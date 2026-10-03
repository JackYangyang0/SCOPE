"""Preserve staged storage when static allocation exceeds the device limit."""
import re
from SCOPE.generate_ir.gpu_resources import hardware_value
from SCOPE.verification.gemm_semantic_checker import extract_launch_config
from SCOPE.verification.cuda_launch_config import configure_shared_launch


def repair_static_shared_overflow(source, ir):
    pattern = re.compile(
        r"(?P<indent>[ \t]*)__shared__\s+float\s+As\[2\]\[BK\]\[BM(?P<apad>\s*\+\s*\d+)?\];\s*"
        r"__shared__\s+float\s+Bs\[2\]\[BK\]\[BN(?P<bpad>\s*\+\s*\d+)?\];"
    )
    declaration = pattern.search(source)
    if not declaration or "cp.async" in source or "extern __shared__" in source:
        return source, []
    c = extract_launch_config(source)
    if not all(type(c.get(k)) is int and c[k] > 0 for k in ("BM", "BN", "BK")):
        return source, []
    a_pad = int(re.search(r"\d+", declaration['apad'] or "0")[0])
    b_pad = int(re.search(r"\d+", declaration['bpad'] or "0")[0])
    a_extent = f"BM + {a_pad}" if a_pad else "BM"
    b_extent = f"BN + {b_pad}" if b_pad else "BN"
    a_offset_extent = f"({a_extent})" if a_pad else a_extent
    size = 2 * c['BK'] * (c['BM'] + a_pad + c['BN'] + b_pad) * 4
    ordinary = hardware_value(ir, 'max_shared_memory_per_block_bytes')
    optin = hardware_value(ir, 'max_shared_memory_per_block_optin_bytes')
    if not ordinary or not ordinary < size <= optin:
        return source, []
    # Require exactly these allocations; additional shared arrays need accounting.
    if source.count('__shared__') != 2:
        return source, []
    indent = declaration['indent']
    replacement = (
        f"{indent}extern __shared__ __align__(16) unsigned char scope_shared[];\n"
        f"{indent}float (&As)[2][BK][{a_extent}] = "
        f"*reinterpret_cast<float (*)[2][BK][{a_extent}]>(scope_shared);\n"
        f"{indent}float (&Bs)[2][BK][{b_extent}] = "
        f"*reinterpret_cast<float (*)[2][BK][{b_extent}]>("
        f"scope_shared + 2 * BK * {a_offset_extent} * sizeof(float));"
    )
    source = source[:declaration.start()] + replacement + source[declaration.end():]
    ir.setdefault('memory', {}).update(shared_memory_allocation='dynamic',
        shared_memory_optin=True, shared_memory_bytes=size)
    ir.setdefault('pipeline', {}).update(stage_count=2, double_buffering=True)
    ir.setdefault('resource', {}).setdefault('shared_memory', {}).update(
        total_bytes=size, pipeline_multiplier=2)
    source = configure_shared_launch(source, ir)
    return source, [{'id': 'STATIC_SHARED_OVERFLOW_OPTIN', 'bytes': size,
                     'static_limit': ordinary, 'optin_limit': optin,
                     'padding': {'A': a_pad, 'B': b_pad},
                     'preserved': ['tile_dimensions', 'two_shared_stages', 'shared_padding', 'float4_loads']}]
