"""Run one independent CUDA pipeline per configured GEMM and publish its winners."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from itertools import count
from pathlib import Path

from SCOPE.tune import bundle_shape_label, load_tuned_candidate, publish_tuned_shape_bundle
from SCOPE.utils.common_utils import load_json, save_json
from SCOPE.utils.redaction import redact


def configured_shapes(values):
    if not isinstance(values, list) or not values:
        raise ValueError("generation.shapes requires a non-empty list of [M, N, K] triples")
    shapes = []
    for value in values:
        if (not isinstance(value, list) or len(value) != 3
                or any(type(n) is not int or n <= 0 for n in value)):
            raise ValueError(f"Expected three positive integers [M, N, K], got {value!r}")
        if value in shapes:
            raise ValueError(f"Duplicate generation shape {value}")
        shapes.append(value)
    return shapes


def new_directory(parent, name):
    parent.mkdir(parents=True, exist_ok=True)
    for n in count(1):
        path = parent / (name if n == 1 else f"{name}-{n}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path


@contextmanager
def batch_lock(directory):
    # An OS lock is released automatically if the coordinating process exits.
    with (directory / ".lock").open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Batch is already running: {directory}") from exc
        yield


def verified_winner(workspace, dimensions):
    shape = dict(zip(("M", "N", "K"), dimensions))
    check = workspace / "results" / "check"
    report = load_json(check / "final_top1_shape_tuning.json")
    if report.get("backend") != "cuda" or len(report.get("shapes", [])) != 1:
        raise ValueError(f"Expected one CUDA Phase 3 result for {shape} in {check}")
    winner = load_tuned_candidate(report, {"strategy_id": "batch"}, shape)
    if winner is None:
        raise ValueError(f"No verified Phase 3 winner for {shape} in {check}")
    source_dir = Path(winner["candidate_code_dir"]).resolve()
    evolution = load_json(check / "evolution_summary.json")
    if (not source_dir.is_relative_to(workspace.resolve() / "gemm_code")
            or evolution.get("best_overall_code_dir") != str(source_dir)
            or evolution.get("best_overall_gflops") != winner["performance"]["gflops"]):
        raise ValueError(f"Export and measured Phase 3 winner disagree in {workspace}")
    item = dict(report["shapes"][0], workspace=str(workspace), verified_ir=str(source_dir / "verified_ir.json"))
    item["kernel_file"] = f"cuda_kernel_{bundle_shape_label(shape)}.cuh"
    item["source_hash"] = load_json(source_dir / "candidate.json")["source_hash"]
    return item, winner["source_snapshot"]


def run_shape_batch(root, config, args, argv):
    settings = config.get("generation") or {}
    shapes = configured_shapes(settings.get("shapes"))
    alias = args.resume_batch or settings.get("alias", "gemm")
    if (not isinstance(alias, str) or not re.fullmatch(r"[\w-][\w.-]{0,63}", alias)
            or alias.endswith(".") or Path(alias).is_reserved()):
        raise ValueError(f"Invalid batch directory name: {alias!r}")
    batches = root / "results" / "batches"
    directory = batches / alias if args.resume_batch else new_directory(batches, alias)
    state_path = directory / "batch.json"
    if args.resume_batch and not state_path.is_file():
        raise FileNotFoundError(f"Batch checkpoint does not exist: {state_path}")
    options = {key: value for key, value in vars(args).items() if key != "resume_batch"}
    options["config"] = redact({k: v for k, v in config.items() if k != "generation"})
    options["template"] = hashlib.sha256(args.template.read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps(options, sort_keys=True, default=str).encode()).hexdigest()
    # The worker receives the normal app CLI with an explicit single-shape override.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--resume-batch")
    _, worker_args = parser.parse_known_args(argv)
    with batch_lock(directory):
        if args.resume_batch:
            state = load_json(state_path)
            if state["dimensions"] != shapes or state["signature"] != signature:
                raise ValueError(f"Shapes, config, template or pipeline options differ from {state_path}")
        else:
            state = {"run": directory.name, "signature": signature, "dimensions": shapes, "shapes": {}}
        state["status"] = "running"
        save_json(state_path, state)
        print(f"SCOPE batch {directory.name}: {directory}", flush=True)
        winners = []
        harness = None
        for dims in shapes:
            label = "x".join(map(str, dims))
            record = state["shapes"].setdefault(label, {"attempts": []})
            try:
                # A process may have exited after saving evolution_summary and before checkpointing.
                attempts = record["attempts"]
                previous = None
                if attempts and record.get("status") != "failed":
                    workspace = directory / attempts[-1]
                    if (workspace / "results/check/evolution_summary.json").is_file():
                        try:
                            previous = verified_winner(workspace, dims)
                        except (OSError, KeyError, ValueError) as exc:
                            print(f"Retrying {label}; previous export could not be verified: {exc}", flush=True)
                if previous is None:
                    workspace = new_directory(directory / label, "run")
                    attempts.append(workspace.relative_to(directory).as_posix())
                    record["status"] = "running"
                    record.pop("gflops", None)
                    save_json(state_path, state)
                    print(f"Generating {label} in {workspace}", flush=True)
                    subprocess.run(
                        [sys.executable, str(root / "app.py"), *worker_args, "--shape-reuse", "off",
                         "--matrix-size", *map(str, dims)],
                        env={**os.environ, "SCOPE_SHAPE_WORKSPACE": str(workspace)}, check=True,
                    )
                    previous = verified_winner(workspace, dims)
                item, sources = previous
                current_harness = {name: sources[name] for name in ("main.cpp", "kernel.h")}
                if harness is not None and harness != current_harness:
                    raise ValueError(f"Incompatible benchmark driver or kernel.h for {label}")
                harness = current_harness
                winners.append(item)
                record.update(status="completed", gflops=item["gflops"])
                record.pop("error", None)
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
                record.update(status="failed", error=str(exc))
                print(f"{label} failed: {exc}", file=sys.stderr, flush=True)
            save_json(state_path, state)
        if len(winners) != len(shapes):
            state["status"] = "incomplete"
            save_json(state_path, state)
            raise RuntimeError(f"Incomplete batch. See {state_path} and retry with --resume-batch {directory.name}")
        bundle = directory / "results"
        try:
            if not bundle.exists():
                with tempfile.TemporaryDirectory(dir=directory, prefix=".publish-") as staging:
                    ready = publish_tuned_shape_bundle(Path(staging), harness, winners, "cuda")
                    ready.rename(bundle)
            else:
                if load_json(bundle / "shape_manifest.json")["shapes"] != winners:
                    raise ValueError(f"Existing manifest differs from the verified winners: {bundle}")
                for item in winners:
                    header = bundle / item["kernel_file"]
                    if hashlib.sha256(header.read_text(encoding="utf-8-sig").encode()).hexdigest() != item["source_hash"]:
                        raise ValueError(f"Published kernel differs from the measured source: {header}")
        except (OSError, KeyError, ValueError) as exc:
            state.update(status="publication_failed", error=str(exc))
            save_json(state_path, state)
            raise
        state.update(status="completed", bundle=str(bundle))
        state.pop("error", None)
        save_json(state_path, state)
        print(f"CUDA bundle: {bundle}", flush=True)
        return bundle
