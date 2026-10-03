"""Idempotent, targeted catalog migration; preserves unrelated user strategies."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from SCOPE.utils.upgrade_gpu_simt_catalog import (
    predicate, strategy, index_summary, find_index_subphase, find_library_subphase, replace_or_append,
)

ROOT = Path(__file__).resolve().parents[1]
GENERAL = "phase2_unlock_general"
SWEEP = "Compiler.ResourceFeedback.PtxasOccupancySweep"


def checks(async_copy=False):
    required = ["dynamic_reference_check", "runtime_safety_check"]
    if async_copy:
        required.append("async_copy_evidence_check")
    return {"verification_stage": "after_code_patch_applied", "constraints": [{
        "constraint_id": "C_ASYNC_PIPELINE" if async_copy else "C_GEMM_RUNTIME",
        "verifiers": {"required": required, "best_effort": ["static_loop_domain_check"]},
    }]}


ASYNC_REQUIREMENTS = [
    "Preserve FP32 SIMT FMA accumulation, alpha/beta, output ownership and existing shared layout; never substitute TF32/FP16.",
    "For each async copy use supported 4/8/16-byte transfers and prove source/destination alignment, or take a synchronous guarded fallback.",
    "Copy into the producer stage only after all consumers of that physical stage have completed; wait for committed copies before reading it.",
    "Use CTA synchronization for data exchanged between warps; a thread-local wait alone is not a CTA-wide producer/consumer barrier.",
    "Handle K-tile counts 0, 1, less than stage_count, exactly stage_count and larger; drain every outstanding group before exit/reuse.",
    "Zero-fill invalid M/N/K lanes without forming invalid global pointers; never read unwritten padding/tail elements.",
    "Keep physical shared shapes, padding, stage_count and auxiliary bytes in IR consistent with the code; replacing 2 stages with 3 allocates 3, not 6.",
    "Materialize the requested stage count as compile-time static __shared__ A/B buffers; Multistage2 is a static double buffer.",
    "Do not introduce extern __shared__, dynamic launch bytes or cudaFuncSetAttribute; dynamic allocation is owned by the separate Memory.SharedMemory.DynamicOptIn strategy.",
    "Use __pipeline_memcpy_async/__pipeline_commit/__pipeline_wait_prior from the fixed cuda_pipeline.h capability include, or equivalent real cp.async PTX; never emit a comment-only marker.",
    "Static evidence checks are not synchronization proofs. Run correctness, CUDA error checks, and compute-sanitizer racecheck/synccheck when available.",
]


def migrate(index, library, graph):
    by_id = {item["strategy_id"]: item for stage in library["stages"]
             for sub in stage["ordered_subphases"] for item in sub["strategies"]}
    records = []
    for count in (2, 3, 4):
        sid = f"Pipeline.CpAsync.Multistage{count}.SharedAB"
        record = copy.deepcopy(by_id.get(sid, by_id["Pipeline.CpAsync.Multistage3.SharedAB"]))
        record.update(strategy_id=sid, canonical_strategy_id=sid,
                      name=f"{count}-stage cp.async shared-memory pipeline",
                      title=f"{count}-stage cp.async shared-memory pipeline")
        record["principle"] = f"Overlap FP32 SIMT computation with asynchronous global-to-shared copies across {count} stages; balance latency hiding against shared-memory residency."
        record["ir_updates"].update({"pipeline.stage_count": count,
                                     "pipeline.double_buffering": count == 2,
                                     "resource.shared_memory.pipeline_multiplier": count})
        record["postconditions"]["patch_ir_verification"]["predicates"] = [
            predicate(f"POST_{i}", field, "eq", value)
            for i, (field, value) in enumerate(record["ir_updates"].items())]
        record["postconditions"]["code_verification"] = checks(True)
        record["code_requirements"] = ASYNC_REQUIREMENTS
        record["validation_cases"] = {"k_tile_counts": [0, 1, count - 1, count, count + 1],
                                      "include_nondivisible_mnk": True, "include_nonzero_beta": True}
        record["resource_effects"]["shared_memory_bytes"] = f"per_stage_bytes * {count} + auxiliary_bytes"
        record["coupled_with"] = []
        record["priority"] = {2: 97, 3: 96, 4: 91}[count]
        records.append(record)
        graph.setdefault("requires", {})[sid] = copy.deepcopy(graph["requires"]["Pipeline.CpAsync.Multistage3.SharedAB"])
        if count == 4:
            graph["requires"][sid].append({"required_state": {"problem.K": ">=1024"}})
        graph.setdefault("conflicts", {})[sid] = [{"severity": "hard", "strategies": [
            "Pipeline.NoAsyncCopy.V1", "Pipeline.DoubleBuffer.*", "Pipeline.WarpAwareDoubleBuffer.*",
            *[f"Pipeline.CpAsync.Multistage{other}.SharedAB" for other in (2, 3, 4) if other != count]]}]
        graph.setdefault("supersedes", {})[sid] = ["Pipeline.NoAsyncCopy.V1", "Pipeline.DoubleBuffer.*", "Pipeline.WarpAwareDoubleBuffer.*"]
        graph.setdefault("fallback", {})[sid] = [{"strategies":
            [f"Pipeline.CpAsync.Multistage{other}.SharedAB" for other in range(count-1, 1, -1)]
            + ["Pipeline.DoubleBuffer.SharedAB", "Pipeline.NoAsyncCopy.V1"]}]

    dynamic = strategy("Memory.SharedMemory.DynamicOptIn", "Layout", "Layout.SharedMemoryTransform",
        "Dynamic shared-memory opt-in", "Materialize the existing layout in dynamic shared memory; configure the actual specialized kernel's allocation at launch. No precision or tiling change.",
        GENERAL, "shared_memory.allocation",
        {"memory.shared_memory_allocation": "dynamic", "memory.shared_memory_optin": True},
        [predicate("PRE_SHARED", "memory.use_shared_memory", "eq", True),
         predicate("PRE_OPTIN", "hardware.supports_shared_memory_optin", "eq", True)],
        ["shared_decl", "launch_config"], ["hardware.supports_shared_memory_optin", "memory.use_shared_memory"],
        94, "medium", {"shared_memory_capacity": "explicit_optin"})
    dynamic["postconditions"]["code_verification"] = checks()
    dynamic["code_requirements"] = [
        "Replace A/B storage with aligned slices of one extern __shared__ allocation; preserve physical shape, padding and stage count.",
        "Report resource.shared_memory.auxiliary_bytes for any non-tile allocations and preserve all consumer indices.",
        "SCOPE supplies the third launch argument and checks cudaFuncSetAttribute; do not insert a second launch or change main.cpp.",
    ]
    records.append(dynamic)
    for percent in (0, 50, 100):
        record = strategy(f"Memory.SharedMemory.Carveout.{percent}", "Compiler", "Compiler.AttributeSelection",
            f"Shared/L1 carveout preference {percent} percent",
            "Request a supported shared/L1 partition preference on the actual kernel, then compare measured correct performance; maximum shared is not always fastest.",
            GENERAL, "shared_memory.carveout", {"memory.shared_memory_carveout_percent": percent},
            [predicate("PRE_SHARED", "memory.use_shared_memory", "eq", True)],
            ["launch_config"], ["memory.use_shared_memory"], 65, "low", {"l1_shared_partition": "tune"})
        record["materialization"] = {"mode": "host_launch_config", "llm_required": False}
        record["postconditions"]["code_verification"] = checks()
        records.append(record)

    for tensor in ("A", "B"):
        sid = f"Layout.SharedMemory.WarpAwareSwizzle{tensor}"
        record = strategy(sid, "Layout", "Layout.SharedMemoryTransform", f"Warp-aware XOR swizzle for {tensor}",
            "Apply a bijective XOR address transform to the shared-memory minor coordinate, using exactly the same transform in producers and consumers. Select mask from lane access pattern and vector alignment, not tensor naming.",
            "phase1_core_coupled", f"layout.shared_{tensor}_transform",
            {f"memory.shared_{tensor}.layout": "xor_swizzle"},
            [predicate("PRE_SHARED", "memory.use_shared_memory", "eq", True),
             predicate("PRE_SHAPE", f"memory.shared_{tensor}.shape", "is_not_null", None)],
            ["shared_decl", "global_to_shared_load", "compute_inner"],
            [f"memory.shared_{tensor}.shape"], 60, "medium", {"bank_conflicts": "measure"})
        record["alias_of"] = f"Layout.SharedMemory.Skew{tensor}.Xor"
        record["canonical_strategy_id"] = record["alias_of"]
        records.append(record)
        canonical = copy.deepcopy(record)
        canonical["strategy_id"] = canonical["canonical_strategy_id"]
        canonical["alias_of"] = None
        canonical["postconditions"]["code_verification"] = checks()
        canonical["code_requirements"] = [
            "Derive and record a bijective swizzle mask from physical row/column extents; do not XOR padded addresses out of allocation bounds.",
            "Preserve vector alignment or scalarize only accesses whose transformed coordinates are not vector-contiguous.",
            "Apply the same map to global-to-shared stores and all shared-to-register loads, including pipeline prologue and epilogue.",
            "Do not claim bank-conflict reduction from a non-null layout field; use measured bank-conflict counters when available.",
        ]
        records.append(canonical)

    record = copy.deepcopy(by_id["Layout.SharedMemory.BankConflictAnalysis"])
    record.update(name="Shared-memory bank-conflict profiling", title="Shared-memory bank-conflict profiling",
                  phase="excluded_default", activation_status="profiler_required", alias_of=None,
                  canonical_strategy_id=record["strategy_id"], conflict_group="analysis.bank_conflicts",
                  modifies_regions=[], provides_fields=[], requires_fields=["memory.shared_A.shape", "memory.shared_B.shape"],
                  risk_level="low")
    record["principle"] = "Analyze per-instruction warp shared addresses, access width and broadcasts, and correlate with hardware bank-conflict counters. This is an analysis pass, not a layout mutation."
    record["preconditions"] = {"predicates": [predicate("PRE_SHARED", "memory.use_shared_memory", "eq", True)]}
    record["postconditions"] = {"patch_ir_verification": {"predicates": []}, "code_verification": checks()}
    record.pop("synthesized_from_index", None)
    records.append(record)

    sweep = copy.deepcopy(by_id[SWEEP])
    sweep.update(phase=GENERAL, activation_status="enabled")
    sweep["name"] = sweep["title"] = "Bounded PTXAS register-budget sweep"
    sweep["principle"] = "Compile the current/default register budget and budgets 64, 96, 128. Record PTXAS registers and spills; retain the fastest correct mean latency. Launch-bounds sweeping is not implemented."
    sweep["materialization"] = {"mode": "compiler_resource_sweep", "llm_required": False}
    sweep["preconditions"] = {"predicates": [predicate("PRE_CORRECT", "verification.correctness.status", "eq", "pass")]}
    sweep["postconditions"]["code_verification"] = checks()
    records.append(sweep)

    for sid in ("Compiler.LaunchBounds.MaxThreads", "Compiler.LaunchBounds.MinBlocksPerSM", "Tuning.WarpOccupancyBalance"):
        record = copy.deepcopy(by_id[sid])
        record.update(name=sid, title=sid, phase="excluded_default", activation_status="executor_required",
                      alias_of=None, canonical_strategy_id=sid, conflict_group="compiler.launch_bounds",
                      modifies_regions=["kernel_declaration"], provides_fields=[],
                      requires_fields=["mapping.threads_per_block"], risk_level="medium")
        record["principle"] = "Tune launch bounds using actual launch thread count, registers, shared bytes and SM residency; require a kernel-declaration edit boundary and measured spills/latency, not merely compiler.attributes != null."
        record["preconditions"] = {"predicates": [predicate("PRE_THREADS", "mapping.threads_per_block", "gt", 0)]}
        record["postconditions"] = {"patch_ir_verification": {"predicates": []}, "code_verification": checks()}
        record["activation_requirements"] = ["kernel-declaration patch anchor", "launch threads <= declared maxThreads",
            "hardware-valid minBlocksPerSM", "compile/register/spill and runtime feedback"]
        record["materialization"] = {"mode": "orchestrator_required", "llm_required": False}
        record.pop("synthesized_from_index", None)
        records.append(record)

    # These need a multi-kernel/context-owning executor; keep them out of local-patch generation.
    for sid in ("Reduction.SplitK.Workspace", "Reduction.StreamK.WorkDecomposition"):
        record = copy.deepcopy(by_id[sid])
        record.update(phase="excluded_default", activation_status="executor_required")
        record.setdefault("name", sid)
        record.setdefault("canonical_strategy_id", sid)
        record.setdefault("alias_of", None)
        record.setdefault("conflict_group", "large_matrix.work_decomposition")
        record.setdefault("modifies_regions", ["launch_config", "index_mapping", "store"])
        record.setdefault("provides_fields", list(record.get("ir_updates", {})))
        record.setdefault("requires_fields", ["problem.K", "tiling.block_k"])
        record["materialization"] = {"mode": "orchestrator_required", "llm_required": False}
        record["activation_requirements"] = ["workspace allocation/lifetime and size limits", "multiple kernel launches on the same stream",
            "exact K partition coverage", "FP32 partial accumulation", "alpha/beta applied once", "workspace+reduction included in measured time"]
        record["selection_policy"] = "Consider low CTA/SM and long-K work as well as large matrices; estimate split overhead before activation."
        record["postconditions"]["code_verification"] = checks()
        records.append(record)

    l2 = strategy("Memory.L2Persistence.AccessWindow", "Pipeline", "Pipeline.LargeMatrixScheduling",
        "L2 persisting access-policy window", "Configure a bounded A or B reuse window and restore prior stream/context policy after use; distinct from CTA swizzling.",
        "excluded_default", "memory.l2_access_window", {"memory.l2_persistence.enabled": True},
        [predicate("PRE_L2", "hardware.supports_l2_persistence", "eq", True)],
        ["launch_config"], ["hardware.max_persisting_l2_cache_bytes", "hardware.max_access_policy_window_bytes"],
        60, "medium", {"l2_setaside": "bounded_by_device_and_reuse"})
    l2["activation_status"] = "executor_required"
    l2["materialization"] = {"mode": "orchestrator_required", "llm_required": False}
    l2["activation_requirements"] = ["own or restore stream access-policy window and context set-aside",
        "respect MIG/MPS restrictions and queried nonzero limits", "include configuration cost in end-to-end report"]
    records.append(l2)

    for record in records:
        subphase = find_index_subphase(index, record["stage"], record["subphase"])
        if record["strategy_id"] not in subphase["allowed_strategy_ids"]:
            subphase["allowed_strategy_ids"].append(record["strategy_id"])
        subphase["candidate_count"] = len(subphase["allowed_strategy_ids"])
        if record["strategy_id"] not in subphase["allowed_strategy_patterns"]:
            subphase["allowed_strategy_patterns"].append(record["strategy_id"])
        replace_or_append(find_library_subphase(library, record["stage"], record["subphase"])["strategies"], record)
        summary = index_summary(record)
        summary["materialization"] = record.get("materialization", {})
        replace_or_append(index["strategies"], summary)
        graph_sub = graph["stages"][record["stage"]]["subphases"][record["subphase"]]
        patterns = graph_sub.setdefault("allowed_strategy_patterns", [])
        if record["strategy_id"] not in patterns:
            patterns.append(record["strategy_id"])
        if record["strategy_id"] not in graph.setdefault("requires", {}):
            graph["requires"][record["strategy_id"]] = []
    index["strategy_count"] = index["strategy_count_total_loaded"] = len(index["strategies"])
    for stage in index["stages"].values():
        for sub in stage["subphases"]:
            index.setdefault("subphase_candidate_summary", {})[sub["subphase_id"]] = {
                "candidate_count": len(sub["allowed_strategy_ids"]), "allowed_strategy_ids": sub["allowed_strategy_ids"]}
    for field, key in (("strategy_ids_by_stage", "stage"), ("strategy_ids_by_category", "category")):
        mapping = {}
        for item in index["strategies"]:
            mapping.setdefault(item[key], []).append(item["strategy_id"])
        index.setdefault("legacy_flat_views", {})[field] = mapping


def main():
    paths = [ROOT / "data/lib/strategy_index.json", ROOT / "data/lib/strategy_library.json",
             ROOT / "data/graph/dependency_graph.json"]
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    docs[2].pop("fallbacks", None)
    migrate(*docs)
    for path, doc in zip(paths, docs):
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
