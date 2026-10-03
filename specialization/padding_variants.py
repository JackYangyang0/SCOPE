"""Conservative padding transforms for directly indexed static shared arrays."""
import re


MODES = {'none': (0, 0), 'a_only': (1, 0), 'ab': (1, 1)}


def padding_variant(source, tile, mode):
    if mode == 'original':
        return source, {}, None
    if mode not in MODES:
        return None, {}, 'unsupported padding mode'
    if re.search(r'float\s*\(&As\)', source):
        from SCOPE.specialization.dynamic_shared import dynamic_padding
        return dynamic_padding(source, tile, mode, padding_variant)
    # Alter only direct multidimensional addressing. Pointer/flat/async copies
    # can embed physical strides that this transform cannot prove equivalent.
    access_source = re.sub(r'^\s*#define.*$', '', source, flags=re.M)
    # C-output casts do not depend on the shared-memory physical layout.
    access_source = re.sub(r'reinterpret_cast<[^>]+>\s*\(\s*&?C\[[^\]]+\]\s*\)', '', access_source)
    # Host-side base-pointer alignment checks do not refer to shared layout.
    access_source = re.sub(r'reinterpret_cast\s*<\s*(?:std::)?uintptr_t\s*>\s*\(\s*[ABC]\s*\)', '', access_source)
    if re.search(r'cp\.async|memcpy_async|reinterpret_cast|FLOAT[24]\s*\(\s*(?:As|Bs)\[', access_source):
        return None, {}, 'pointer/vector/async shared access requires a separate coupled implementation'
    edits, metadata, total = [], {}, 0
    for name, pad in zip(('As', 'Bs'), MODES[mode]):
        pattern = rf'__shared__\s+float\s+{name}\s*\[(\d+)\]\s*\[(BK|BM)\]\s*\[((?:BM|BN|BK))(?:\s*\+\s*(\d+))?\]\s*;'
        matches = list(re.finditer(pattern, source))
        if len(matches) != 1:
            return None, {}, f'{name}: unsupported or ambiguous declaration'
        match = matches[0]
        stages, outer, inner, _ = match.groups()
        # Every non-declaration occurrence must use all three array indices.
        body = source[:match.start()] + source[match.end():]
        for use in re.finditer(rf'\b{name}\b', body):
            if not re.match(r'\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\[[^\]]+\]', body[use.end():]):
                return None, {}, f'{name}: non-direct shared-array use'
        extent = inner + (' + 1' if pad else '')
        edits.append((match.start(), match.end(), f'__shared__ float {name}[{stages}][{outer}][{extent}];'))
        shape = [int(stages), tile[outer], tile[inner] + pad]
        metadata['shared_' + name[0]] = {'padding': pad, 'shape': shape}
        total += 4 * shape[0] * shape[1] * shape[2]
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    metadata['shared_memory_bytes'] = total
    return source, metadata, None
