"""Conservative source checks for known SCOPE optimization regressions.

These checks detect supported source forms; they are not a CUDA semantic proof.
"""
import re


def active_source(source):
    source = re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", "", source)
    return re.sub(r"(?m)^\s*#define[^\n]*", "", source)


def vector_load_widths(source):
    """Recognize direct A/B vector read expressions, not pointer declarations.

    This is source evidence only, not proof of alignment or tile coverage.
    """
    code = active_source(source)
    result = {}
    for operand in ("A", "B"):
        for width in (2, 4):
            patterns = (
                rf"=\s*FLOAT{width}\s*\(\s*{operand}\s*\[",
                rf"=\s*\*\s*reinterpret_cast\s*<\s*(?:const\s+)?float{width}\s*\*\s*>\s*"
                rf"\(\s*(?:&\s*{operand}\s*\[|{operand}\s*\+)",
            )
            if any(re.search(pattern, code) for pattern in patterns):
                result.setdefault(operand, set()).add(width)
    return result


def vector_loads(source):
    return set(vector_load_widths(source))


def staged_buffers(source):
    code = active_source(source)
    constants = dict(re.findall(r'\b(?:constexpr|const)\s+(?:unsigned\s+)?int\s+(\w+)\s*=\s*([234])\s*;', code))
    result = {}
    # Recognize the exact non-overlapping, typed opt-in allocation emitted by
    # shared_storage_repair. Arbitrary pointer aliases are not evidence.
    if 'extern __shared__ __align__(16) unsigned char scope_shared[];' in code:
        a = re.search(
            r'float\s*\(&As\)\[2\]\[BK\]\[(?P<extent>BM(?:\s*\+\s*\d+)?)\]\s*=\s*'
            r'\*reinterpret_cast<float\s*\(\*\)\[2\]\[BK\]\[(?P=extent)\]>\(scope_shared\);',
            code,
        )
        b = re.search(
            r'float\s*\(&Bs\)\[2\]\[BK\]\[(?P<extent>BN(?:\s*\+\s*\d+)?)\]\s*=\s*'
            r'\*reinterpret_cast<float\s*\(\*\)\[2\]\[BK\]\[(?P=extent)\]>\('
            r'scope_shared\s*\+\s*2\s*\*\s*BK\s*\*\s*\(?(?P<a_extent>BM(?:\s*\+\s*\d+)?)\)?\s*'
            r'\*\s*sizeof\(float\)\);',
            code,
        )
        if a and b and re.sub(r'\s+', '', a['extent']) == re.sub(r'\s+', '', b['a_extent']):
            result.update(As=2, Bs=2)
    for name, stages in re.findall(
        r"__shared__\s+float\s+(\w+)\s*\[\s*(\w+)\s*\]\s*\[[^]]+\]\s*\[", code):
        value = constants.get(stages, stages)
        if value in ('2', '3', '4'):
            result[name] = int(value)
    return result


def preservation_defects(before, after, strategy):
    sid = strategy.get("strategy_id", "")
    defects = []
    # A named scalar/single-buffer strategy may intentionally replace a feature.
    # Repair is never an implicit permission to downgrade.
    scalar_transition = sid.startswith("Vectorization.") and "Scalar" in sid
    if not scalar_transition:
        for operand in sorted(vector_loads(before) - vector_loads(after)):
            defects.append(f"vectorized {operand} load removed; use an explicit fallback candidate")
        before_widths, after_widths = vector_load_widths(before), vector_load_widths(after)
        for operand in sorted(before_widths.keys() & after_widths.keys()):
            explicit_width_change = sid.startswith(f"Vectorization.GlobalLoad{operand}.") or sid.startswith("Vectorization.GlobalLoadAB.")
            if not explicit_width_change and max(after_widths[operand]) < max(before_widths[operand]):
                defects.append(f"vectorized {operand} load width reduced; use an explicit fallback candidate")
    single_transition = False  # NoAsyncCopy means synchronous copy, not single buffering.
    if not single_transition:
        after_buffers = staged_buffers(after)
        for name, stages in staged_buffers(before).items():
            if after_buffers.get(name, 0) < stages:
                defects.append(f"shared buffer {name} lost {stages}-stage storage; use an explicit fallback candidate")
    return defects
