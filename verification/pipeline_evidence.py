"""Advisory pipeline realization evidence, not a CUDA race-freedom proof."""
import re
from SCOPE.verification.optimization_preservation import active_source


def pipeline_evidence(source):
    code = active_source(source)
    async_copy = 'cp.async' in code or 'memcpy_async' in code
    double = bool(re.search(r'\b(?:As|Bs)\s*\[2\]', code))
    loop = re.search(r'for\s*\(int\s+bkIdx\s*=\s*1;[^)]*\)', code)
    serialized = False
    compute_load_barrier = False
    if loop:
        body = code[loop.end():]
        load = re.search(r'\b(?:As|Bs)\[mem_flag\][^;]*=', body)
        barrier = body.find('__syncthreads()')
        compute = re.search(r'for\s*\(int\s+k\s*=\s*0;\s*k\s*<\s*BK;', body)
        serialized = bool(load and compute and load.start() < barrier < compute.start())
        compute_load_barrier = bool(load and compute and compute.start() < load.start() < barrier)
    return {'double_buffer_storage': double, 'async_copy_present': async_copy,
            'load_barrier_compute_pattern': serialized,
            'compute_load_barrier_pattern': compute_load_barrier,
            'overlap_proven': False,
            'message': 'Buffer count is not evidence of load/compute overlap. Inspect generated instructions and measure.'}


PIPELINE_OBLIGATIONS = [
    'Two shared buffers alone do not establish a latency-hiding pipeline. Do not claim overlap for load -> barrier -> compute.',
    'Compute -> synchronous next-tile load -> barrier also has no explicit prefetch overlap. To implement overlap, couple producer scheduling, register/shared staging, waits and buffer-reuse safety; do not merely allocate a second buffer.',
    'For cp.async, keep actual copy/commit/wait operations; never repair by deleting the selected async strategy.',
    'A stored as As[stage][k][m] is transposed relative to row-major global A: a contiguous 16-byte A copy cannot directly scatter four K components into it. Use a compatible staged layout with consistent consumers, or a valid staged transpose.',
    'For each stage track produced K tile, committed group, wait before consumption and barrier before buffer reuse. Handle prologue, steady state and drain with exactly-once K coverage.',
    'Never simply remove __syncthreads to improve overlap. Prove producer/consumer ordering and cross-thread reuse before altering synchronization.',
]
