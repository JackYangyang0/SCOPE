"""Conservative parent-relative no-op detection, not general C++ equivalence."""
import hashlib
import json
import re

from SCOPE.verification.cuda_build_options import cuda_optimization_flags


def normalized_source(source):
    # Location/time macros, raw strings and line splicing need preprocessing.
    if re.search(r'__LINE__|__FILE__|__DATE__|__TIME__|R"|\\\r?\n', source):
        return None
    tokens = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/'
    def replace(match):
        text = match[0]
        return ' ' + '\n' * text.count('\n') if text.startswith(('//', '/*')) else text
    source = re.sub(tokens, replace, source)
    return '\n'.join(line.strip() for line in source.splitlines() if line.strip())


def implementation_fingerprint(snapshot, ir, platform):
    files = {}
    for name, source in sorted(snapshot.items()):
        value = normalized_source(source)
        if value is None:
            return None
        files[name] = value
    if not files:
        return None
    payload = {'files': files, 'flags': cuda_optimization_flags(ir), 'platform': platform,
               'problem': ir.get('problem', {}), 'hardware': ir.get('hardware', {})}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def unchanged_verified_parent(parent_ir, parent_source, ir, source, platform):
    verification = parent_ir.get('verification', {})
    compile_report = verification.get('compile', {})
    if not verification.get('accepted') or compile_report.get('status') != 'pass':
        return None
    if compile_report.get('effective_optimization_flags') != cuda_optimization_flags(parent_ir):
        return None
    if compile_report.get('build_platform') != platform:
        return None
    parent = implementation_fingerprint(parent_source, parent_ir, platform)
    current = implementation_fingerprint(source, ir, platform)
    return current if current and parent == current else None
