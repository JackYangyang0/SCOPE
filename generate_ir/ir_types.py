"""Normalize declared scalar IR fields, never executable expressions or unknown fields."""
import math
import re


INTEGER_PATHS = {
    *(f"tiling.{name}" for name in ("block_m", "block_n", "block_k", "thread_m", "thread_n")),
    *(f"tiling.warp_tile.{name}" for name in ("warp_m", "warp_n", "warp_m_iter", "warp_n_iter")),
    *(f"mapping.{name}" for name in ("warps_m", "warps_n", "warps_per_block", "threads_per_block",
                                    "block_dim_x", "block_dim_y", "block_dim_z", "outputs_per_thread")),
    "resource.shared_memory.pipeline_multiplier", "resource.shared_memory.total_bytes",
    "resource.shared_memory.per_stage_bytes", "resource.register.estimated_per_thread",
    *(f"vectorization.{operand}.{field}" for operand in "ABC"
      for field in ("vector_width", "alignment_required_bytes")),
}
BOOLEAN_PATHS = {
    "vectorization.enabled", "memory.use_shared_memory", "memory.use_register_tile",
    *(f"vectorization.{operand}.{field}" for operand in "ABC" for field in
      ("vectorized_load", "vectorized_store", "alignment_guard", "alignment_proven", "tail_handling")),
}


def integer_value(value, path):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise ValueError(f"Invalid IR type at {path}: expected integer, got {value!r}")


def normalize_ir_types(data, prefix=""):
    """Copy modified branches only; support both nested IR and dotted patch updates."""
    if prefix in INTEGER_PATHS:
        return None if data is None else integer_value(data, prefix)
    if prefix in BOOLEAN_PATHS:
        if data is None or isinstance(data, bool):
            return data
        if isinstance(data, str) and data.strip().lower() in ("true", "false"):
            return data.strip().lower() == "true"
        raise ValueError(f"Invalid IR type at {prefix}: expected boolean, got {data!r}")
    if not isinstance(data, dict):
        if data is not None and any(p.startswith(prefix + ".") for p in INTEGER_PATHS | BOOLEAN_PATHS):
            raise ValueError(f"Invalid IR type at {prefix}: expected object, got {data!r}")
        return data
    result = dict(data)
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if path in INTEGER_PATHS | BOOLEAN_PATHS or any(p.startswith(path + ".") for p in INTEGER_PATHS | BOOLEAN_PATHS):
            result[key] = normalize_ir_types(value, path)
    return result
