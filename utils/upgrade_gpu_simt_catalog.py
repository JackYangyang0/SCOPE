from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data" / "lib" / "strategy_index.json"
LIBRARY_PATH = ROOT / "data" / "lib" / "strategy_library.json"
GRAPH_PATH = ROOT / "data" / "graph" / "dependency_graph.json"
GRAPH_METADATA_PATH = ROOT / "data" / "graph" / "dependency_graph_metadata.json"
OBSOLETE_STRATEGY_IDS = {"Tiling.WarpTile.Iteration.WMITERxWNITER"}


def predicate(check_id: str, field: str, op: str, value: Any) -> dict[str, Any]:
    if op in {"not_null", "nonnull"}:
        op = "is_not_null"
    elif op == "null":
        op = "is_null"
    return {
        "id": check_id,
        "kind": "ir_predicate",
        "op": op,
        "lhs": {"field": field},
        "rhs": {"const": value},
    }


def strategy(
    strategy_id: str,
    stage: str,
    subphase: str,
    title: str,
    principle: str,
    phase: str,
    conflict_group: str,
    ir_updates: dict[str, Any],
    preconditions: list[dict[str, Any]],
    modifies_regions: list[str],
    requires_fields: list[str],
    priority: int,
    risk: str,
    resource_effects: dict[str, str],
) -> dict[str, Any]:
    anchor_names = {
        "launch_config": "LAUNCH_CONFIG",
        "tile_indexing": "INDEX_MAPPING",
        "index_mapping": "INDEX_MAPPING",
        "register_file": "REGISTER_DECL",
        "register_decl": "REGISTER_DECL",
        "shared_decl": "SHARED_DECL",
        "global_to_shared_load": "GLOBAL_TO_SHARED_LOAD",
        "main_loop": "MAIN_LOOP",
        "compute_inner": "COMPUTE_INNER",
        "sync_after_load": "SYNC_AFTER_LOAD",
        "store": "STORE",
        "build_config": "BUILD_CONFIG",
    }
    post_predicates = [
        predicate(f"POST_{index}", field, "eq", value)
        for index, (field, value) in enumerate(ir_updates.items(), start=1)
    ]
    return {
        "strategy_id": strategy_id,
        "stage": stage,
        "category": stage,
        "subphase": subphase,
        "name": title,
        "title": title,
        "principle": principle,
        "preconditions": {
            "verification_stage": "before_patch_generation",
            "input": ["OptIR_t"],
            "verifier": "ir_predicate_eval",
            "predicates": preconditions,
        },
        "postconditions": {
            "patch_ir_verification": {
                "verification_stage": "after_patch_json_before_code_application",
                "input": ["OptIR_t", "Patch.ir_updates"],
                "verifier": "ir_predicate_eval",
                "predicates": post_predicates,
            },
            "code_verification": {
                "verification_stage": "after_code_patch_applied",
                "constraints": [],
            },
        },
        "ir_updates": ir_updates,
        "affected_fields": sorted(ir_updates),
        "allowed_modified_regions": list(dict.fromkeys(anchor_names[region] for region in modifies_regions)),
        "resource_effects": resource_effects,
        "risk_level": risk,
        "priority": priority,
        "maturity": "v1",
        "phase": phase,
        "canonical_strategy_id": strategy_id,
        "alias_of": None,
        "conflict_group": conflict_group,
        "modifies_regions": modifies_regions,
        "provides_fields": sorted(ir_updates),
        "requires_fields": requires_fields,
        "materialization": {
            "mode": "llm_structured_patch",
            "llm_required": True,
            "scope": "local_anchor_regions_only",
        },
    }


NEW_STRATEGIES = [
    strategy(
        "Tiling.ThreadTile.4x8",
        "Tiling",
        "Tiling.ThreadTileSelection",
        "Per-thread 4x8 rectangular register tile",
        "Increase accumulator reuse along N while retaining a moderate M extent; the stage controller derives warp iteration fields from the selected warp tile.",
        "phase1_core_coupled",
        "tiling.thread_tile",
        {"tiling.thread_m": 4, "tiling.thread_n": 8, "memory.use_register_tile": True},
        [
            predicate("PRE_WARP_M", "tiling.warp_tile.warp_m", "not_null", None),
            predicate("PRE_WARP_N", "tiling.warp_tile.warp_n", "not_null", None),
        ],
        ["launch_config", "tile_indexing", "register_file"],
        ["tiling.warp_tile.warp_m", "tiling.warp_tile.warp_n", "hardware.warp_size"],
        82,
        "medium",
        {"register_pressure": "increase", "data_reuse": "increase", "instruction_level_parallelism": "increase"},
    ),
    strategy(
        "Tiling.ThreadTile.8x4",
        "Tiling",
        "Tiling.ThreadTileSelection",
        "Per-thread 8x4 rectangular register tile",
        "Increase accumulator reuse along M while retaining a moderate N extent; this complements tall CTA and warp tiles.",
        "phase1_core_coupled",
        "tiling.thread_tile",
        {"tiling.thread_m": 8, "tiling.thread_n": 4, "memory.use_register_tile": True},
        [
            predicate("PRE_WARP_M", "tiling.warp_tile.warp_m", "not_null", None),
            predicate("PRE_WARP_N", "tiling.warp_tile.warp_n", "not_null", None),
        ],
        ["launch_config", "tile_indexing", "register_file"],
        ["tiling.warp_tile.warp_m", "tiling.warp_tile.warp_n", "hardware.warp_size"],
        82,
        "medium",
        {"register_pressure": "increase", "data_reuse": "increase", "instruction_level_parallelism": "increase"},
    ),
    strategy(
        "Reordering.ComputeBundle.RegisterTiledOuterProduct",
        "MappingReordering",
        "Reordering.ComputeSchedule",
        "Register-tiled outer-product compute bundle",
        "Couple A/B register fragments, K-major loading, an outer-product FFMA schedule, and bounded unrolling so the generated loop exposes independent FP32 FFMA instructions without omitting the K remainder.",
        "phase1_core_coupled",
        "schedule.compute_bundle",
        {
            "register.cache_A": True,
            "register.cache_B": True,
            "schedule.ffma": "outer_product_latency_hidden",
            "schedule.warp_compute": "k_outer_inner",
            "schedule.unrolling": "bounded_by_block_k",
        },
        [
            predicate("PRE_SHARED", "memory.use_shared_memory", "eq", True),
            predicate("PRE_REGISTER_TILE", "memory.use_register_tile", "eq", True),
            predicate("PRE_TM", "tiling.thread_m", "not_null", None),
            predicate("PRE_TN", "tiling.thread_n", "not_null", None),
        ],
        ["register_decl", "compute_inner", "main_loop"],
        ["memory.use_shared_memory", "memory.use_register_tile", "tiling.thread_m", "tiling.thread_n"],
        94,
        "medium",
        {"register_pressure": "increase", "ffma_issue_rate": "increase", "shared_load_reuse": "increase"},
    ),
    strategy(
        "Pipeline.CpAsync.Multistage3.SharedAB",
        "Pipeline",
        "Pipeline.BufferingSelection",
        "Three-stage cp.async shared-memory pipeline",
        "Use cp.async to overlap global-to-shared transfer with register-tiled computation across three rotating shared-memory stages, including commit/wait ordering and a drained epilogue.",
        "phase2_unlock_general",
        "pipeline.buffering",
        {
            "pipeline.enabled": True,
            "pipeline.async_copy": True,
            "pipeline.copy_mechanism": "cp_async",
            "pipeline.stage_count": 3,
            "pipeline.double_buffering": False,
            "resource.shared_memory.pipeline_multiplier": 3,
        },
        [
            predicate("PRE_CP_ASYNC", "hardware.supports_cp_async", "eq", True),
            predicate("PRE_SHARED", "memory.use_shared_memory", "eq", True),
            predicate("PRE_SHARED_BYTES", "resource.shared_memory.total_bytes", "not_null", None),
        ],
        ["shared_decl", "global_to_shared_load", "main_loop", "sync_after_load"],
        ["hardware.supports_cp_async", "memory.use_shared_memory", "resource.shared_memory.total_bytes"],
        96,
        "high",
        {"shared_memory_bytes": "triple", "global_load_latency": "hide", "occupancy": "may decrease"},
    ),
    strategy(
        "Pipeline.CpAsync.Multistage4.SharedAB",
        "Pipeline",
        "Pipeline.BufferingSelection",
        "Four-stage cp.async shared-memory pipeline",
        "Use a four-stage cp.async ring buffer for long K reductions when the additional shared-memory footprint still permits useful CTA residency.",
        "phase2_unlock_general",
        "pipeline.buffering",
        {
            "pipeline.enabled": True,
            "pipeline.async_copy": True,
            "pipeline.copy_mechanism": "cp_async",
            "pipeline.stage_count": 4,
            "pipeline.double_buffering": False,
            "resource.shared_memory.pipeline_multiplier": 4,
        },
        [
            predicate("PRE_CP_ASYNC", "hardware.supports_cp_async", "eq", True),
            predicate("PRE_SHARED", "memory.use_shared_memory", "eq", True),
            predicate("PRE_SHARED_BYTES", "resource.shared_memory.total_bytes", "not_null", None),
            predicate("PRE_K", "problem.K", "ge", 1024),
        ],
        ["shared_decl", "global_to_shared_load", "main_loop", "sync_after_load"],
        ["hardware.supports_cp_async", "memory.use_shared_memory", "resource.shared_memory.total_bytes", "problem.K"],
        91,
        "high",
        {"shared_memory_bytes": "quadruple", "global_load_latency": "hide", "occupancy": "may decrease"},
    ),
    strategy(
        "Scheduling.PersistentCTA.GridStrideTiles",
        "Pipeline",
        "Pipeline.LargeMatrixScheduling",
        "Persistent CTA grid-stride tile scheduler",
        "Launch a bounded resident CTA set and let each CTA consume multiple output tiles through a grid-stride tile loop; no direct SMID dependency is required.",
        "phase2_unlock_large_matrix",
        "large_matrix.schedule",
        {
            "scheduling.persistent_cta": "grid_stride_tiles",
            "scheduling.tile_assignment": "grid_stride",
        },
        [
            predicate("PRE_SM_COUNT", "hardware.sm_count", "ge", 1),
            predicate("PRE_BM", "tiling.block_m", "not_null", None),
            predicate("PRE_BN", "tiling.block_n", "not_null", None),
        ],
        ["launch_config", "index_mapping", "main_loop"],
        ["hardware.sm_count", "tiling.block_m", "tiling.block_n", "mapping.threads_per_block"],
        88,
        "high",
        {"launch_overhead": "decrease", "wave_quantization": "improve", "l2_reuse": "may increase"},
    ),
    strategy(
        "Reduction.SplitK.Workspace",
        "Pipeline",
        "Pipeline.LargeMatrixScheduling",
        "Workspace Split-K parallel reduction",
        "Partition long K reductions across CTAs, write FP32 partial tiles to a workspace, and launch a deterministic final reduction kernel.",
        "phase2_unlock_large_matrix",
        "large_matrix.work_decomposition",
        {
            "scheduling.work_decomposition": "split_k_workspace",
            "reduction.workspace_required": True,
            "reduction.finalization": "separate_kernel",
        },
        [
            predicate("PRE_K", "problem.K", "ge", 2048),
            predicate("PRE_BK", "tiling.block_k", "not_null", None),
        ],
        ["launch_config", "index_mapping", "store"],
        ["problem.K", "tiling.block_k", "mapping.threads_per_block"],
        84,
        "high",
        {"parallelism": "increase_for_large_k", "workspace_bytes": "increase", "reduction_overhead": "increase"},
    ),
    strategy(
        "Compiler.ResourceFeedback.PtxasOccupancySweep",
        "Compiler",
        "Compiler.ResourceTuning",
        "PTXAS resource and occupancy feedback sweep",
        "Compile a bounded set of launch-bounds and max-register variants, parse PTXAS register/shared-memory usage, and retain the fastest correct resource point.",
        "phase2_unlock_general",
        "compiler.resource_tuning",
        {
            "compiler.resource_feedback.enabled": True,
            "compiler.resource_feedback.metric": "correct_gflops",
            "compiler.resource_feedback.source": "ptxas_and_runtime",
        },
        [],
        ["build_config", "launch_config"],
        [],
        92,
        "low",
        {"compile_variants": "increase", "occupancy": "tune", "register_pressure": "tune"},
    ),
]


for _strategy_id, _bm, _bn, _bk in [
    ("Tiling.BlockTile.32x128x16", 32, 128, 16),
    ("Tiling.BlockTile.64x32x16", 64, 32, 16),
    ("Tiling.BlockTile.128x32x16", 128, 32, 16),
    ("Tiling.BlockTile.64x64x32", 64, 64, 32),
    ("Tiling.BlockTile.64x128x32", 64, 128, 32),
    ("Tiling.BlockTile.128x64x32", 128, 64, 32),
    ("Tiling.BlockTile.128x128x32", 128, 128, 32),
]:
    NEW_STRATEGIES.append(
        strategy(
            _strategy_id,
            "Tiling",
            "Tiling.BlockTileSelection",
            f"CTA tile {_bm}x{_bn}x{_bk}",
            "Choose a concrete CTA output tile and K-slice depth; downstream warp/thread tiling and resource checks determine final viability.",
            "phase1_core_coupled",
            "tiling.block_tile",
            {"tiling.enabled": True, "tiling.block_m": _bm, "tiling.block_n": _bn, "tiling.block_k": _bk},
            [
                predicate("PRE_M", "problem.M", "not_null", None),
                predicate("PRE_N", "problem.N", "not_null", None),
                predicate("PRE_K", "problem.K", "not_null", None),
            ],
            ["launch_config"],
            ["problem.M", "problem.N", "problem.K"],
            80,
            "medium" if _bk == 32 else "low",
            {
                "cta_work": "increase" if max(_bm, _bn) >= 128 else "moderate",
                "shared_memory_bytes": "increase" if _bk == 32 else "moderate",
            },
        )
    )

for _strategy_id, _wm, _wn in [
    ("Tiling.WarpTile.16x16", 16, 16),
    ("Tiling.WarpTile.16x64", 16, 64),
    ("Tiling.WarpTile.64x16", 64, 16),
]:
    NEW_STRATEGIES.append(
        strategy(
            _strategy_id,
            "Tiling",
            "Tiling.WarpTileSelection",
            f"Warp tile {_wm}x{_wn}",
            "Partition the selected CTA tile into a concrete warp output tile while preserving exact divisibility and warp-level output coverage.",
            "phase1_core_coupled",
            "tiling.warp_tile",
            {"tiling.warp_tile.warp_m": _wm, "tiling.warp_tile.warp_n": _wn},
            [
                predicate("PRE_BM", "tiling.block_m", "not_null", None),
                predicate("PRE_BN", "tiling.block_n", "not_null", None),
                predicate("PRE_WARP", "hardware.warp_size", "not_null", None),
            ],
            ["launch_config", "tile_indexing"],
            ["tiling.block_m", "tiling.block_n", "hardware.warp_size"],
            79,
            "low",
            {"warps_per_block": "shape_dependent", "warp_output_reuse": "shape_dependent"},
        )
    )

for _strategy_id, _tm, _tn in [
    ("Tiling.ThreadTile.2x4", 2, 4),
    ("Tiling.ThreadTile.4x2", 4, 2),
    ("Tiling.ThreadTile.2x8", 2, 8),
    ("Tiling.ThreadTile.8x2", 8, 2),
]:
    NEW_STRATEGIES.append(
        strategy(
            _strategy_id,
            "Tiling",
            "Tiling.ThreadTileSelection",
            f"Per-thread {_tm}x{_tn} register tile",
            "Choose an asymmetric per-thread output fragment; the controller derives WMITER/WNITER and validates complete warp coverage.",
            "phase1_core_coupled",
            "tiling.thread_tile",
            {"tiling.thread_m": _tm, "tiling.thread_n": _tn, "memory.use_register_tile": True},
            [
                predicate("PRE_WARP_M", "tiling.warp_tile.warp_m", "not_null", None),
                predicate("PRE_WARP_N", "tiling.warp_tile.warp_n", "not_null", None),
                predicate("PRE_WARP", "hardware.warp_size", "not_null", None),
            ],
            ["launch_config", "tile_indexing", "register_file"],
            ["tiling.warp_tile.warp_m", "tiling.warp_tile.warp_n", "hardware.warp_size"],
            83 if _tm * _tn <= 8 else 81,
            "low" if _tm * _tn <= 8 else "medium",
            {"register_pressure": "moderate", "data_reuse": "increase", "shape_bias": "asymmetric"},
        )
    )


# These strategies need runtime orchestration outside cuda_kernel.cuh. Keep them
# documented and dependency-checked, but do not expose them to normal selection.
for _record in NEW_STRATEGIES:
    if _record["strategy_id"] in {
        "Reduction.SplitK.Workspace",
        "Compiler.ResourceFeedback.PtxasOccupancySweep",
    }:
        _record["phase"] = "excluded_default"
        _record["activation_status"] = "executor_required"
        _record["materialization"] = {
            "mode": "orchestrator_required",
            "llm_required": False,
            "scope": "runtime_or_build_orchestration",
        }


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_index_subphase(index: dict[str, Any], stage_id: str, subphase_id: str) -> dict[str, Any]:
    stage = index["stages"][stage_id]
    return next(item for item in stage["subphases"] if item["subphase_id"] == subphase_id)


def find_library_subphase(library: dict[str, Any], stage_id: str, subphase_id: str) -> dict[str, Any]:
    stage = next(item for item in library["stages"] if item["stage_id"] == stage_id)
    return next(item for item in stage["ordered_subphases"] if item["subphase"] == subphase_id)


def replace_or_append(items: list[dict[str, Any]], record: dict[str, Any]) -> None:
    strategy_id = record["strategy_id"]
    for index, item in enumerate(items):
        if item.get("strategy_id") == strategy_id:
            items[index] = record
            return
    items.append(record)


def remove_obsolete_strategies(value: Any) -> None:
    """Remove retired strategy records and references from catalog documents."""
    if isinstance(value, dict):
        for key in list(value):
            if key in OBSOLETE_STRATEGY_IDS:
                del value[key]
                continue
            remove_obsolete_strategies(value[key])
    elif isinstance(value, list):
        value[:] = [
            item
            for item in value
            if not (isinstance(item, str) and item in OBSOLETE_STRATEGY_IDS)
            and not (isinstance(item, dict) and item.get("strategy_id") in OBSOLETE_STRATEGY_IDS)
        ]
        for item in value:
            remove_obsolete_strategies(item)


def index_summary(record: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "strategy_id": record["strategy_id"],
        "stage": record["stage"],
        "category": record["category"],
        "subphase": record["subphase"],
        "name": record["name"],
        "short_description": record["principle"],
        "intent": record["principle"],
        "risk_summary": record["risk_level"],
        "priority": record["priority"],
        "maturity": record["maturity"],
        "phase": record["phase"],
        "canonical_strategy_id": record["canonical_strategy_id"],
        "alias_of": record["alias_of"],
        "conflict_group": record["conflict_group"],
        "modifies_regions": record["modifies_regions"],
        "provides_fields": record["provides_fields"],
        "requires_fields": record["requires_fields"],
    }
    if record.get("activation_status"):
        summary["activation_status"] = record["activation_status"]
    return summary


def update_catalog() -> None:
    index = load(INDEX_PATH)
    library = load(LIBRARY_PATH)
    graph = load(GRAPH_PATH)
    metadata = load(GRAPH_METADATA_PATH)

    remove_obsolete_strategies(index)
    remove_obsolete_strategies(library)
    remove_obsolete_strategies(graph)
    remove_obsolete_strategies(metadata)

    index["index_version"] = "8.1-joint-tiling"
    library["model_version"] = "8.1"
    graph["graph_version"] = "8.1"
    metadata["graph_version"] = "8.1"

    for record in NEW_STRATEGIES:
        if record["strategy_id"].startswith(("Tiling.BlockTile.", "Tiling.WarpTile.", "Tiling.ThreadTile.")):
            record["materialization"] = {
                "mode": "deterministic",
                "template": "launch_config_constants",
                "llm_required": False,
                "scope": "local_anchor_region_or_ir_only",
            }
        subphase = find_index_subphase(index, record["stage"], record["subphase"])
        if record["strategy_id"] not in subphase["allowed_strategy_ids"]:
            subphase["allowed_strategy_ids"].append(record["strategy_id"])
        prefix = record["strategy_id"].rsplit(".", 1)[0] + ".*"
        if prefix not in subphase["allowed_strategy_patterns"]:
            subphase["allowed_strategy_patterns"].append(prefix)
        subphase["candidate_count"] = len(subphase["allowed_strategy_ids"])
        subphase["provides_fields"] = sorted(set(subphase.get("provides_fields", [])) | set(record["provides_fields"]))

        library_subphase = find_library_subphase(library, record["stage"], record["subphase"])
        replace_or_append(library_subphase["strategies"], record)
        replace_or_append(index["strategies"], index_summary(record))

    index["strategies"].sort(key=lambda item: item["strategy_id"])
    index["strategy_count_total_loaded"] = len(index["strategies"])
    index["strategy_count"] = len(index["strategies"])

    by_stage: dict[str, list[str]] = {}
    by_category: dict[str, list[str]] = {}
    for item in index["strategies"]:
        by_stage.setdefault(item["stage"], []).append(item["strategy_id"])
        by_category.setdefault(item["category"], []).append(item["strategy_id"])
    index["legacy_flat_views"]["strategy_ids_by_stage"] = by_stage
    index["legacy_flat_views"]["strategy_ids_by_category"] = by_category

    for stage in index["stages"].values():
        for subphase in stage.get("subphases", []):
            ids = subphase.get("allowed_strategy_ids", [])
            subphase["candidate_count"] = len(ids)
            index["subphase_candidate_summary"][subphase["subphase_id"]] = {
                "candidate_count": len(ids),
                "allowed_strategy_ids": ids,
            }

    graph_pipeline = graph["stages"]["Pipeline"]
    buffering = graph_pipeline["subphases"]["Pipeline.BufferingSelection"]
    for pattern in ["Pipeline.CpAsync.Multistage*.SharedAB"]:
        if pattern not in buffering["allowed_strategy_patterns"]:
            buffering["allowed_strategy_patterns"].append(pattern)
    if "Pipeline.LargeMatrixScheduling" not in graph_pipeline["subphase_order"]:
        verification_index = graph_pipeline["subphase_order"].index("Pipeline.StageVerification")
        graph_pipeline["subphase_order"].insert(verification_index, "Pipeline.LargeMatrixScheduling")
    large_matrix_subphase = graph_pipeline["subphases"].setdefault(
        "Pipeline.LargeMatrixScheduling",
        {
            "allowed_strategy_patterns": [
                "Mapping.CTASwizzle.*",
                "Memory.L2Reuse.*",
                "Scheduling.PersistentCTA.*",
                "Reduction.StreamK.*",
            ],
            "requires_fields": [
                "problem.size_class",
                "tiling.block_m",
                "tiling.block_n",
                "mapping.threads_per_block",
            ],
            "provides_fields": [
                "scheduling.cta_swizzle",
                "memory.l2_reuse_policy",
                "scheduling.persistent_cta",
                "scheduling.work_decomposition",
            ],
            "optional": True,
            "next": ["Pipeline.StageVerification"],
        },
    )
    graph_pipeline["subphases"]["Pipeline.PrefetchSelection"]["next"] = [
        "Pipeline.LargeMatrixScheduling"
    ]
    large_matrix_subphase["allowed_strategy_patterns"].extend(
        pattern
        for pattern in ["Reduction.SplitK.*"]
        if pattern not in large_matrix_subphase["allowed_strategy_patterns"]
    )
    compute_patterns = graph["stages"]["MappingReordering"]["subphases"]["Reordering.ComputeSchedule"][
        "allowed_strategy_patterns"
    ]
    if "Reordering.ComputeBundle.*" not in compute_patterns:
        compute_patterns.append("Reordering.ComputeBundle.*")

    graph["requires"] = {
        "Reordering.ComputeBundle.RegisterTiledOuterProduct": [
            {
                "mode": "all",
                "strategies": ["Tiling.ThreadTile.*", "Layout.SharedMemory.AB.Basic"],
            },
            {"required_state": {"memory.use_register_tile": True, "mapping.thread_to_output": "not null"}},
        ],
        "Pipeline.CpAsync.Multistage3.SharedAB": [
            {
                "mode": "all",
                "strategies": ["Layout.SharedMemory.AB.Basic", "Reordering.LoadCompute.SeparatePhases"],
            },
            {"required_state": {"hardware.supports_cp_async": True, "memory.use_shared_memory": True}},
        ],
        "Pipeline.CpAsync.Multistage4.SharedAB": [
            {
                "mode": "all",
                "strategies": ["Layout.SharedMemory.AB.Basic", "Reordering.LoadCompute.SeparatePhases"],
            },
            {"required_state": {"hardware.supports_cp_async": True, "memory.use_shared_memory": True, "problem.K": ">=1024"}},
        ],
        "Scheduling.PersistentCTA.GridStrideTiles": [
            {"required_state": {"tiling.block_m": "not null", "tiling.block_n": "not null", "hardware.sm_count": ">=1"}}
        ],
        "Reduction.SplitK.Workspace": [
            {"mode": "all", "strategies": ["Tiling.BlockTile.*"]},
            {"required_state": {"problem.K": ">=2048", "tiling.block_k": "not null"}},
        ],
        "Compiler.ResourceFeedback.PtxasOccupancySweep": [
            {"required_state": {"verification.compile.status": "pass"}}
        ],
    }
    graph["conflicts"] = {
        "Reordering.ComputeBundle.RegisterTiledOuterProduct": [
            {
                "severity": "hard",
                "strategies": ["Register.FFMA.Schedule*", "Reordering.WarpCompute.*"],
            }
        ],
        "Pipeline.CpAsync.Multistage3.SharedAB": [
            {
                "severity": "hard",
                "strategies": ["Pipeline.NoAsyncCopy.V1", "Pipeline.DoubleBuffer.*", "Pipeline.WarpAwareDoubleBuffer.*", "Pipeline.CpAsync.Multistage4.SharedAB"],
            }
        ],
        "Pipeline.CpAsync.Multistage4.SharedAB": [
            {
                "severity": "hard",
                "strategies": ["Pipeline.NoAsyncCopy.V1", "Pipeline.DoubleBuffer.*", "Pipeline.WarpAwareDoubleBuffer.*", "Pipeline.CpAsync.Multistage3.SharedAB"],
            }
        ],
        "Scheduling.PersistentCTA.GridStrideTiles": [
            {"severity": "hard", "strategies": ["Scheduling.PersistentCTA.StaticTileLoop", "Reduction.StreamK.*", "Reduction.SplitK.*"]}
        ],
        "Reduction.SplitK.Workspace": [
            {"severity": "hard", "strategies": ["Reduction.StreamK.*", "Scheduling.PersistentCTA.*"]}
        ],
    }
    graph["fallback"] = {
        "Tiling.ThreadTile.4x8": [{"strategies": ["Tiling.ThreadTile.4x4", "Tiling.ThreadTile.2x2"]}],
        "Tiling.ThreadTile.8x4": [{"strategies": ["Tiling.ThreadTile.4x4", "Tiling.ThreadTile.2x2"]}],
        "Tiling.ThreadTile.2x4": [{"strategies": ["Tiling.ThreadTile.2x2", "Tiling.ThreadTile.4x4"]}],
        "Tiling.ThreadTile.4x2": [{"strategies": ["Tiling.ThreadTile.2x2", "Tiling.ThreadTile.4x4"]}],
        "Tiling.ThreadTile.2x8": [{"strategies": ["Tiling.ThreadTile.2x4", "Tiling.ThreadTile.2x2"]}],
        "Tiling.ThreadTile.8x2": [{"strategies": ["Tiling.ThreadTile.4x2", "Tiling.ThreadTile.2x2"]}],
        "Reordering.ComputeBundle.RegisterTiledOuterProduct": [
            {"strategies": ["Register.FFMA.ScheduleOuterProduct", "Reordering.KLoop.Unroll4"]}
        ],
        "Pipeline.CpAsync.Multistage4.SharedAB": [
            {"strategies": ["Pipeline.CpAsync.Multistage3.SharedAB", "Pipeline.DoubleBuffer.SharedAB", "Pipeline.NoAsyncCopy.V1"]}
        ],
        "Pipeline.CpAsync.Multistage3.SharedAB": [
            {"strategies": ["Pipeline.DoubleBuffer.SharedAB", "Pipeline.NoAsyncCopy.V1"]}
        ],
        "Scheduling.PersistentCTA.GridStrideTiles": [
            {"strategies": ["Mapping.CTASwizzle.GroupedN", "Mapping.CTASwizzle.GroupedM"]}
        ],
        "Reduction.SplitK.Workspace": [{"strategies": ["Reduction.StreamK.WorkDecomposition"]}],
        "Compiler.ResourceFeedback.PtxasOccupancySweep": [
            {"strategies": ["Tuning.WarpOccupancyBalance", "Compiler.Ptxas.MaxRegisterCount"]}
        ],
    }
    graph["supersedes"] = {
        "Pipeline.CpAsync.Multistage3.SharedAB": [
            "Pipeline.NoAsyncCopy.V1",
            "Pipeline.DoubleBuffer.*",
            "Pipeline.WarpAwareDoubleBuffer.*",
        ],
        "Pipeline.CpAsync.Multistage4.SharedAB": [
            "Pipeline.NoAsyncCopy.V1",
            "Pipeline.DoubleBuffer.*",
            "Pipeline.WarpAwareDoubleBuffer.*",
        ],
        "Scheduling.PersistentCTA.GridStrideTiles": ["Scheduling.PersistentCTA.StaticTileLoop"],
        "Reduction.SplitK.Workspace": ["Reduction.StreamK.*"],
    }
    graph["phase_flow"] = {
        "phase1": [
            "Tiling",
            "Layout",
            "MappingReordering",
            "Vectorization",
            "Pipeline",
            "Epilogue",
            "Compiler",
            "FinalVerification",
            "Phase1Top3",
        ],
        "phase2": [
            "MatrixProfile",
            "UnlockCandidateFilter",
            "LLMBatchPlan",
            "BatchPatchApply",
            "CompileCorrectnessPerformance",
            "RepairFallbackRollback",
            "UnifiedTop3Ranking",
        ],
        "entry_condition": "Phase1Top3 compile, correctness, and runtime_safety pass",
    }

    metadata["core_principle"] = (
        "Stage-oriented construction followed by size-aware performance unlock. "
        "Performance strategies carry explicit IR prerequisites, hard conflicts, and rollback fallbacks."
    )
    metadata["global_flow"] = [
        "Spec",
        "OptIR_Init",
        "Phase1_StageController",
        "TilingStage",
        "LayoutStage",
        "MappingReorderingStage",
        "VectorizationStage",
        "PipelineStage",
        "EpilogueStage",
        "CompilerStage",
        "Phase1Verification",
        "Phase1Top3",
        "MatrixProfile",
        "Phase2BatchUnlock",
        "UnifiedTop3Ranking",
        "AcceptKernel",
    ]

    save(INDEX_PATH, index)
    save(LIBRARY_PATH, library)
    save(GRAPH_PATH, graph)
    save(GRAPH_METADATA_PATH, metadata)


if __name__ == "__main__":
    update_catalog()
