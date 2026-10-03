"""Detect a missing producer barrier in the recognized staged GEMM protocol."""
import re
import hashlib
from pathlib import Path


def inspect_initial_shared_sync(source):
    from SCOPE.verification.load_transform import mask_comments
    code = mask_comments(source)
    markers = list(re.finditer(r'/\*\s*\*?\s*MAIN_LOOP_BEGIN\s*\*/', source))
    loads = list(re.finditer(r'/\*\s*\*?\s*GLOBAL_TO_SHARED_LOAD_BEGIN\s*\*/', source))
    kernel = re.search(r'__global__[\s\S]*?\bvoid\s+gemm\s*\([^)]*\)\s*\{', code)
    base = {'id': 'GEMM_INITIAL_SHARED_PRODUCER_BARRIER', 'status': 'pass',
            'message': 'Initial-load barrier check not applicable to this source form.',
            'detail': {'recognized': False}}
    if len(markers) != 1 or len(loads) != 1 or not kernel:
        return base
    marker, load = markers[0], loads[0]
    if not kernel.end() < load.start() < marker.start():
        return base
    prefix = code[kernel.end():marker.start()]
    if (prefix.count('{') != prefix.count('}')
            or re.search(r'\b(?:return|goto)\b|cp\.async|memcpy_async', prefix)
            or not prefix.rstrip().endswith((';', '}'))):
        return base
    stores = list(re.finditer(r'\b(?:As|Bs)\[[^;=]+\]\s*=|FLOAT4\(\s*Bs\[[^;]+?\)\s*=',
                              code[load.end():marker.start()]))
    read = re.search(r'=\s*(?:As|Bs)\[', code[marker.end():])
    if not stores or not read:
        return base
    last_store = load.end() + stores[-1].end()
    first_read = marker.end() + read.start()
    between = code[last_store:first_read]
    base['detail']['recognized'] = True
    # This is a narrow absence check, not a proof that arbitrary conditional
    # barriers dominate reads. Racecheck remains necessary on executed inputs.
    if '__syncthreads' in between:
        base['message'] = 'A block barrier is present between initial stores and first shared read.'
        return base
    base.update(status='fail', message='No block barrier between cooperative initial shared stores and first shared read.',
                failure_type='GEMM.Semantic.MissingInitialSharedBarrier',
                repair_action='Insert an unconditional block barrier after completed initial loads, before the main loop.')
    base['detail']['insertion_offset'] = marker.end()
    return base


def repair_initial_shared_sync(source):
    report = inspect_initial_shared_sync(source)
    if report['status'] != 'fail':
        return source, []
    offset = report['detail']['insertion_offset']
    return source[:offset] + '\n    __syncthreads(); // Publish the initial cooperative tile.\n' + source[offset:], [report]


def prepare_initial_shared_sync(chain_dir):
    """Repair before terminal compilation, preserving a source-hashed checkpoint."""
    from SCOPE.utils.common_utils import save_json
    kernel = Path(chain_dir) / 'cuda_kernel.cuh'
    if not kernel.exists():
        return None
    before = kernel.read_text(encoding='utf-8')
    after, changes = repair_initial_shared_sync(before)
    if not changes:
        return None
    digest = hashlib.sha256(before.encode()).hexdigest()
    checkpoint = kernel.parent / ('initial_shared_sync_repair.' + digest[:12] + '.json')
    report = {'method': 'deterministic_initial_producer_barrier', 'changes': changes,
              'source_sha256_before': digest, 'source_sha256_after': hashlib.sha256(after.encode()).hexdigest(),
              'checkpoint': str(checkpoint), 'requires_compile_run_verification': True}
    save_json(checkpoint, {**report, 'source_before': before})
    kernel.write_text(after, encoding='utf-8')
    return report
