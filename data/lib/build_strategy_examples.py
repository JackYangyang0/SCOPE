"""Build audited per-strategy references; fail instead of inventing a fallback.

Run from the repository parent: python -m SCOPE.data.lib.build_strategy_examples
Only example metadata and generated reference data are updated.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def nodes(value):
    if isinstance(value, dict):
        if isinstance(value.get('strategy_id'), str):
            yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child)


def spec(code, conditions, obligations, kind='adaptable_code_fragment', notes=()):
    return dict(code_lines=code.strip().splitlines(), applicability=list(conditions),
                verification_obligations=list(obligations), kind=kind,
                adaptation_rules=list(notes))


def gpu(s):
    if s.startswith('Tiling.'):
        category = s.split('.')[1]
        names = {'BlockTile': ('BM','BN','BK'), 'WarpTile': ('WM','WN'), 'ThreadTile': ('TM','TN')}[category]
        values = s.split('.')[-1].split('x')
        values = values if len(values) == len(names) and all(v.isdigit() for v in values) else ['IR_'+n for n in names]
        return spec('\n'.join(f'constexpr int {n} = {v};' for n,v in zip(names, values)),
                    ['Parameter declarations belong in cuda_gemm, not duplicated in gemm',
                     'Resolve symbolic IR_* values from the selected strategy/IR, never invent them'],
                    ['Exactly one declaration per parameter', 'Recompute compatible warp/lane mapping and resources after all tile parameters are available'],
                    'deterministic_parameter_example', ['The executor applies these parameters; do not introduce an LLM code-generation call.'])
    if s.startswith('Mapping.LaneLayout.'):
        mode = s.rsplit('.',1)[1]
        dims = {'1DContiguousM':('32','1','lane','0'), '1DContiguousN':('1','32','0','lane'),
                '2D_8x4':('8','4','lane / 4','lane % 4')}[mode]
        return spec(f'const int lane = linear_tid % 32;\nconst int lane_m = {dims[2]};\nconst int lane_n = {dims[3]};',
                    [f'This example defines lane_m extent {dims[0]} and lane_n extent {dims[1]}', 'Update compute and store coordinates consistently'],
                    ['All 32 lanes have unique coordinates', 'Selected warp tile must fit lane extents and per-lane fragments'],
                    notes=['8x4 means 8 M coordinates and 4 N coordinates here; check the selected mapping contract, not the label alone.'])
    if s == 'Mapping.Warp.BasicWidLane':
        return spec('const int linear_tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);\nconst int wid = linear_tid / 32;\nconst int lane = linear_tid % 32;',
                    ['CUDA warp size 32'], ['Launch thread count matches mapping', 'Do not flatten a 2D launch with threadIdx.x alone'])
    if s.startswith(('Mapping.Warp.OutputFragment', 'Mapping.WarpThreadTile.')):
        return spec('''const int warp_n_count = BN / WN;
const int warp_m = wid / warp_n_count, warp_n = wid % warp_n_count;
const int lane_cols = WNITER / TN;
const int lane_m = lane / lane_cols, lane_n = lane % lane_cols;
for (int mi = 0; mi < WM / WMITER; ++mi)
  for (int ni = 0; ni < WN / WNITER; ++ni) {
    const int m = warp_m * WM + mi * WMITER + lane_m * TM;
    const int n = warp_n * WN + ni * WNITER + lane_n * TN;
    // The lane owns [m,m+TM) x [n,n+TN); use the same mapping in compute/store.
  }''', ['BM%WM==0, BN%WN==0, WM%WMITER==0, WN%WNITER==0',
           'WMITER%TM==0, WNITER%TN==0, (WMITER/TM)*(WNITER/TN)==32', 'These WMITER/WNITER are fragment extents, not iteration counts'],
          ['Each output has exactly one owner', 'Full BM x BN coverage without out-of-range shared indices'])
    if s == 'Layout.SharedMemory.AB.Basic':
        return spec('__shared__ float As[STAGES][BK][BM];\n__shared__ float Bs[STAGES][BK][BN];',
                    ['FP32, A stored as [K][M], B as [K][N]', 'STAGES comes from the selected pipeline, not necessarily 2'],
                    ['All allocated elements initialized before use', 'Shared-memory bytes include all stages and padding'])
    if s == 'Layout.SharedMemory.BankConflictAnalysis':
        return spec('''word_addr = physical_shared_byte_address // 4
bank = word_addr % 32
# Group addresses per warp instruction, accounting for broadcasts and vector transactions.
# Confirm suspected conflicts with profiler counters before choosing padding/swizzle.''',
                    ['32-bank, four-byte bank-width model; validate against target architecture'],
                    ['Analyze producer and consumer access patterns', 'Do not declare conflicts from variable names', 'Do not mutate code merely to report analysis'], 'diagnostic_recipe')
    if '.Padding' in s:
        operands = 'AB' if 'PaddingAB' in s else ('A' if 'PaddingA' in s else 'B')
        code = '\n'.join(f'__shared__ float {t}s[STAGES][BK][B{"M" if t=="A" else "N"} + 1];' for t in operands)
        return spec(code, ['Example assumes A[K][M], B[K][N]', 'Keep logical extents unchanged; pad the actual physical minor axis'],
                    ['Every consumer and pointer cast uses padded strides', 'Resource count includes padding', 'Recheck vector alignment: +1 row stride may break float4 shared stores'])
    if any(x in s for x in ('.SkewA.', '.SkewB.', '.WarpAwareSwizzleA', '.WarpAwareSwizzleB')):
        t = 'A' if ('SkewA' in s or s.endswith('A')) else 'B'
        return spec(f'''const int physical_col = logical_col ^ (logical_row & XOR_MASK);
{t}s[stage][logical_row][physical_col] = incoming;
// Consumer uses the identical bijection:
float value = {t}s[stage][logical_row][logical_col ^ (logical_row & XOR_MASK)];''',
                    ['Minor extent is a power of two; XOR_MASK only uses bits within that extent', 'For contiguous float4 groups the transform must preserve or explicitly handle the low two bits'],
                    ['Prove bijection and bounds for the full tile', 'Use identical mapping in loads and compute', 'Measure bank conflicts; XOR alone does not prove improvement'])
    if '.Transpose' in s:
        t = s[-1]
        code = ('As[stage][k][m] = A[global_m * K + global_k];\nfloat a = As[stage][k][m];' if t=='A'
                else 'Bs[stage][n][k] = B[global_k * N + global_n];\nfloat b = Bs[stage][n][k];')
        return spec(code, ['Global operands remain row-major; transpose the shared layout only'],
                    [f'Declare {t}s with matching transposed physical dimensions', 'Update every load, compute, vector cast and buffer stage', 'Scalar scatter and contiguous cp.async are not interchangeable'])
    if s.startswith('Layout.SharedMemory.VectorizedStore'):
        t = 'A' if 'StoreA' in s else 'B'
        return spec(f'*reinterpret_cast<float4*>(&{t}s[stage][row][col]) = incoming_vec;',
                    ['Four components have the same shared row and consecutive physical columns', 'Destination is 16-byte aligned including padding and stage stride'],
                    ['No shared-row crossing', 'Transposed row-major A K components cannot be packed across M', 'Retain scalar scatter when the selected layout forbids this store and report the contract incompatibility'])
    if s.startswith(('Register.AccumulatorLayout.', 'Layout.RegisterTile.')):
        if s.endswith('FlatArray'):
            code = 'float acc[ROWS * COLS] = {0};\nacc[i * COLS + j] = fmaf(a, b, acc[i * COLS + j]);'
        elif s.endswith('ScalarUnrolled'):
            code = 'float c00 = 0, c01 = 0, c10 = 0, c11 = 0;\nc00 = fmaf(a0,b0,c00); c01 = fmaf(a0,b1,c01);\nc10 = fmaf(a1,b0,c10); c11 = fmaf(a1,b1,c11);'
        else:
            code = 'float acc[ROWS][COLS] = {0};\nacc[i][j] = fmaf(a, b, acc[i][j]);'
        return spec(code, ['ROWS/COLS are total per-thread ownership including warp-fragment iterations'],
                    ['Initialize every accumulator once before the full K reduction', 'Store each accumulator with identical indexing', 'Scalar example is 2x2: expand to actual ownership, not just TM/TN blindly'])
    if s == 'Reordering.LoadCompute.SeparatePhases':
        return spec('''for (int k0 = 0; k0 < K; k0 += BK) {
  LOAD_CURRENT_TILE(k0); // guarded cooperative loads, zero-fill inactive K
  __syncthreads();
  COMPUTE_CURRENT_TILE();
  __syncthreads(); // finish readers before overwriting the same storage
}''', ['Synchronous load/compute phases'], ['Preserve selected buffering and vectorization when adapting', 'No barrier in thread-divergent guards', 'Materialize helper bodies'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith('Reordering.ThreadMapping.CoalescedLoad') or s == 'Reordering.WarpSharedLoad.StripMined':
        t = 'B' if s.endswith('LoadB') else 'A'
        count, cols = ('BK * BN','BN') if t=='B' else ('BM * BK','BK')
        return spec(f'''for (int q = linear_tid; q < {count}; q += threads) {{
  int row = q / {cols}, col = q % {cols};
  // Adjacent lanes load adjacent global columns; scatter into the selected shared layout.
  LOAD_AND_SCATTER_{t}(row, col);
}}''', ['This scalar task schedule is a coordinate illustration; expand tasks in vector units when vector width > 1'],
                    ['Complete coverage when tile tasks exceed thread count', 'Guard global boundaries', 'For warp-shared strip mining partition warp tasks disjointly before lane iteration'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith('Register.Cache'):
        t = 'A' if 'CacheA' in s else 'B'
        return spec(f'''#pragma unroll
for (int i = 0; i < FRAGMENT; ++i) reg{t}[i] = SHARED_{t}(stage, k, i);
// Reuse reg{t} over the other output dimension before advancing k.''',
                    ['Resolve FRAGMENT and SHARED accessor from actual ownership/layout'], ['Initialize all lanes before FMA use', 'No stale values across k/stage changes', 'Track register pressure'], 'algorithm_sketch_with_explicit_placeholders')
    if s in ('Register.FFMA.ScheduleInterleaved','Register.FFMA.ScheduleOuterProduct','Reordering.WarpCompute.KOuterInnerSchedule','Reordering.ComputeBundle.RegisterTiledOuterProduct'):
        return spec('''for (int k = 0; k < valid_k; ++k) {
  LOAD_A_B_FRAGMENTS(k);
  for (int i = 0; i < ROWS; ++i)
    for (int j = 0; j < COLS; ++j)
      acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
}''', ['Each accumulator is independent across output positions, dependent across K'],
                    ['Do not omit warp-fragment iterations', 'For interleaving alternate independent accumulators, not dependent updates to one accumulator', 'Materialize fragment loads; preserve operand layout'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith('Reordering.KLoop.'):
        mode = s.rsplit('.',1)[1]
        if mode in ('Unroll4','Unroll8','ManualUnroll','UnrollMicroK'):
            u = {'Unroll4':'4','Unroll8':'8','ManualUnroll':'IR_UNROLL','UnrollMicroK':'IR_MICRO_K'}[mode]
            code = f'''int k = 0;
for (; k + {u} <= valid_k; k += {u}) {{
  #pragma unroll
  for (int u = 0; u < {u}; ++u) ACCUMULATE_K(k + u);
}}
for (; k < valid_k; ++k) ACCUMULATE_K(k);'''
        else:
            pragma = '#pragma unroll 1' if mode=='NoUnrollForRegisterPressure' else '#pragma unroll'
            code = pragma+'\nfor (int k = 0; k < valid_k; ++k) ACCUMULATE_K(k);'
        return spec(code, ['Resolve valid_k and unroll factors from IR; BK-full unroll needs a compile-time bound'],
                    ['Each K term exactly once', 'Remainder handling retained', 'Check ptxas spills and code size'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith('Safety.') or s == 'Vectorization.AlignmentGuard':
        return safety(s)
    if s.startswith(('Reordering.WarpCooperativeLoad', 'Vectorization.GlobalLoad')):
        t = 'A' if ('LoadA.' in s) else 'B'
        width = 2 if s.endswith('float2') else 4
        tasks,cols = ('BM','BK') if t=='A' else ('BK','BN')
        return spec(f'''for (int v = worker; v < {tasks} * ({cols} / {width}); v += workers) {{
  const int row = v / ({cols}/{width}), col = (v % ({cols}/{width})) * {width};
  float{width} value = *reinterpret_cast<const float{width}*>(GLOBAL_{t}_ADDRESS(row,col));
  SCATTER_{t}_COMPONENTS(row,col,value);
}}''', [f'Global {t} contiguous minor dimension, {width*4}-byte alignment', 'worker/workers is the disjoint CTA or warp partition selected by IR'],
                    ['Tasks per vector, not per scalar', 'Guard partial vectors and row tails', 'Shared component mapping matches layout'], 'algorithm_sketch_with_explicit_placeholders')
    if s == 'Pipeline.NoAsyncCopy.V1':
        return spec('''LOAD_TILE_WITH_SYNCHRONOUS_LOADS(stage, k0);
__syncthreads();
COMPUTE_TILE(stage);''', ['No cp.async instructions'], ['NoAsyncCopy does NOT mean single buffering', 'Preserve selected stage count and protect reuse'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith('Memory.Prefetch.GlobalToRegister') or s == 'Pipeline.WarpRegisterPrefetchAB':
        which = 'AB' if s.endswith('AB') else s[-1]
        return spec(f'''PREFETCH_{which}_REGISTERS(0);
for (int k = 0; k < valid_k; ++k) {{
  if (k + 1 < valid_k) PREFETCH_{which}_REGISTERS_INTO_NEXT_SLOT(k + 1);
  COMPUTE_WITH_CURRENT_FRAGMENTS(k);
  SWAP_CURRENT_AND_NEXT_REGISTER_SLOTS();
}}''', ['Independent current/next register slots, correct global/shared source for the selected strategy'],
                    ['First prefetch before use, no final out-of-range read', 'Never overwrite live operands', 'Preserve warp mapping and track register spills'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith(('Mapping.CTASwizzle.', 'Memory.L2Reuse.CTASwizzle')):
        nfirst = s.endswith('GroupedN')
        major,minor = ('tiles_n','tiles_m') if nfirst else ('tiles_m','tiles_n')
        return spec(f'''const int pid = blockIdx.x + blockIdx.y * gridDim.x;
const int group = pid / (GROUP * {minor});
const int first = group * GROUP;
const int extent = min(GROUP, {major} - first);
const int local = pid % (GROUP * {minor});
const int major_tile = first + local % extent;
const int minor_tile = local / extent;
// {'tile_n=major_tile, tile_m=minor_tile' if nfirst else 'tile_m=major_tile, tile_n=minor_tile'}''',
                    ['Positive GROUP; grid launches exactly tiles_m*tiles_n CTAs', 'GroupedN here groups N coordinates; adapt to the selected graph convention'],
                    ['Bijection including last partial group', 'Do not increase grid extent accidentally', 'Measure L2 reuse, not just swizzle presence'])
    if s.startswith('Scheduling.PersistentCTA.'):
        return spec('''const int tiles = tiles_m * tiles_n;
for (int task = blockIdx.x; task < tiles; task += gridDim.x) {
  const int tile_m = task / tiles_n, tile_n = task % tiles_n;
  RESET_ACCUMULATORS_AND_PIPELINE();
  COMPUTE_AND_STORE_ONE_TILE(tile_m, tile_n);
  __syncthreads(); // complete readers before CTA-local storage reuse
}''', ['1D persistent grid bounded using measured residency', 'Uniform loop trip count within each CTA'],
                    ['Tile ownership is disjoint and exhaustive', 'Reset all per-tile state', 'Do not assume persistent scheduling improves every shape'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith('Reduction.'):
        return reduction(s)
    if s == 'Memory.L2Persistence.AccessWindow':
        return spec('''cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.base_ptr = operand;
attr.accessPolicyWindow.num_bytes = window_bytes;
attr.accessPolicyWindow.hitRatio = hit_ratio;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
// Check cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside_bytes).
// Check cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr).
// Launch on that stream, then restore its window and reset persisting lines when appropriate.''',
                    ['Device supports persisting L2, access window and set-aside sizes within probed limits'],
                    ['Check API return values', 'No stale pointer/window after allocation lifetime', 'Restore caller stream policy'], notes=['Not supported uniformly across modes such as MIG; do not force unsupported attributes.'])
    if s.startswith(('Epilogue.', 'Mapping.WarpStore.', 'Vectorization.StoreC.')):
        return epilogue(s)
    if s.startswith(('Compiler.', 'Memory.SharedMemory.Carveout.', 'Scheduling.WaveQuantization.', 'Tuning.')):
        return compiler(s)
    raise ValueError('No curated GPU reference for '+s)


def safety(s):
    guarded = ('GeneralGuarded','MaskLoad')
    if any(s.endswith(x) for x in guarded):
        code = 'As[stage][k][m] = (gm < M && gk < K) ? A[gm*K+gk] : 0.0f;\nBs[stage][k][n] = (gk < K && gn < N) ? B[gk*N+gn] : 0.0f;\n__syncthreads(); // outside element predicates'
    elif s.endswith('PaddingInput'):
        code = 'MP = ceil_div(M,BM)*BM; NP = ceil_div(N,BN)*BN; KP = ceil_div(K,BK)*BK;\nALLOCATE_AND_ZERO_PADDED_ABC(MP,NP,KP);\nCOPY_LOGICAL_AB_WITH_PADDED_STRIDES();\nif (beta != 0) COPY_LOGICAL_C_TO_PADDED_C();\nRUN_KERNEL_ON_PADDED_STORAGE_WITH_PADDED_LEADING_DIMENSIONS();\nCOPY_BACK_ONLY_LOGICAL_C();\nRELEASE_WORKSPACE();'
    elif s.endswith('SeparateTailKernel'):
        code = 'RUN_FULL_TILE_KERNEL(M/BM, N/BN);\nRUN_TAIL_KERNEL_FOR_OUTPUTS_OUTSIDE_FULL_RECTANGLE();\n// Each region computes the complete K reduction including its K tail.'
    elif s.endswith('ScalarFallback'):
        code = 'if (fast_shape_and_alignment_proven) optimized_gemm(...);\nelse scalar_gemm_full_reduction(...); // exactly one path executes'
    elif s.endswith('StaticDivisibleNoGuard') or s.endswith('AssumeDivisibleAligned'):
        code = 'const bool valid = M%BM==0 && N%BN==0 && K%BK==0 && K%VECTOR_WIDTH==0 && N%VECTOR_WIDTH==0\n    && reinterpret_cast<uintptr_t>(A)%(4*VECTOR_WIDTH)==0\n    && reinterpret_cast<uintptr_t>(B)%(4*VECTOR_WIDTH)==0\n    && reinterpret_cast<uintptr_t>(C)%(4*STORE_WIDTH)==0 && N%STORE_WIDTH==0;\n// Enforce valid at specialization/dispatch boundary before entering unguarded kernel.'
    else:
        code = 'const bool aligned = reinterpret_cast<uintptr_t>(address) % (sizeof(float)*VECTOR_WIDTH) == 0;\nif (aligned && all_components_in_bounds) VECTOR_ACCESS();\nelse GUARDED_SCALAR_COMPONENT_ACCESSES();'
    return spec(code, ['Use the selected boundary policy; this example does not authorize a different fallback strategy'],
                ['No out-of-bounds pointer formation/access', 'Invalid loaded K values zero-filled', 'No omitted/duplicated output or beta application', 'Dispatch/guard is executable, not a comment'], 'algorithm_sketch_with_explicit_placeholders')


def reduction(s):
    if 'SplitK' in s:
        code = '''const int split = blockIdx.z;
const int begin = split * K / SPLITS, end = (split + 1) * K / SPLITS;
// Each split computes only [begin,end) including tile remainders.
workspace[(split * M + m) * N + n] = partial_sum;
// Separate ordered reduction kernel:
float sum = 0;
for (int p = 0; p < SPLITS; ++p) sum += workspace[(p * M + m) * N + n];
C[m*N+n] = alpha * sum + (beta == 0 ? 0 : beta * C[m*N+n]);'''
    else:
        code = '''const long long total = (long long)tiles_m * tiles_n * k_tiles;
const long long begin = worker * total / workers, end = (worker + 1) * total / workers;
for (long long pos = begin; pos < end;) {
  const long long tile = pos / k_tiles;
  const long long stop = min(end, (tile + 1) * k_tiles);
  COMPUTE_PARTIAL_TILE(tile, pos % k_tiles, stop - tile * k_tiles);
  WRITE_UNIQUE_PARTIAL_SLOT(worker, tile);
  pos = stop;
}
// Ordered reduction combines all partials for each output tile; alpha/beta once.'''
    return spec(code, ['Workspace and reduction launch are part of the implementation', 'Use checked 64-bit indexing/allocation arithmetic for large workloads'],
                ['Reduction ranges partition the entire K domain', 'No cross-CTA spin barrier assuming all CTAs are resident', 'No partial slot collisions', 'Workspace lifetime and synchronization correct', 'Benchmark GEMM plus reduction, not just partial GEMM'], 'algorithm_sketch_with_explicit_placeholders')


def epilogue(s):
    if '.Fusion.' in s:
        op = s.rsplit('.',1)[1]
        expr = {'Bias':'z + bias[n]', 'ReLU':'fmaxf(z, 0.0f)', 'GELU':'0.5f*z*(1.0f+erff(z*0.7071067811865475f))'}[op]
        return spec(f'float z = alpha * acc + (beta == 0 ? 0 : beta * C[idx]);\nC[idx] = {expr};',
                    ['Fusion explicitly requested by GEMM spec', 'Bias axis and GELU exact/approximate formula match the requested semantics'],
                    ['Apply fusion once after full K reduction', 'Reference checker uses same operation order', 'Guard output bounds'])
    if any(x in s for x in ('float2','float4','AlignedNoGuard','GuardedVectorStore')):
        w = 2 if 'float2' in s else 4
        code = f'''if (m < M && n + {w-1} < N && reinterpret_cast<uintptr_t>(&C[m*N+n]) % {4*w} == 0) {{
  float{w} out = GATHER_ADJACENT_OUTPUTS_AND_APPLY_ALPHA_BETA(m,n);
  *reinterpret_cast<float{w}*>(&C[m*N+n]) = out;
}} else {{
  STORE_VALID_COMPONENTS_SCALAR(m,n);
}}'''
        return spec(code, [f'One thread owns {w} adjacent logical N outputs', 'Use IR vector width for parametric policies'],
                    ['Gather actual adjacent accumulators; no unsafe array reinterpretation', 'Handle beta==0 without reading C', 'Do not overlap stores across lanes', 'For AlignedNoGuard remove branches only with executable specialization proof'], 'algorithm_sketch_with_explicit_placeholders')
    expr = 'alpha * acc' if 'BetaZero' in s else ('alpha * acc + C[idx]' if 'BetaOne' in s else 'alpha * acc + (beta == 0 ? 0 : beta * C[idx])')
    return spec(f'if (global_m < M && global_n < N) {{\n  const size_t idx = (size_t)global_m * N + global_n;\n  C[idx] = {expr};\n}}',
                ['BetaZero requires beta==0; BetaOne requires beta==1 when selected', 'Each lane/thread output ownership comes from current mapping'],
                ['Global rather than block-local boundary guards', 'Coalesced variants map adjacent lanes to adjacent N outputs without changing ownership', 'Alpha/beta applied exactly once'])


def compiler(s):
    exact = {
        'Compiler.FastMath.Enabled': ('nvcc --use_fast_math ...', 'FP32 tolerance permits changed division, sqrt, denormal and contraction behavior'),
        'Compiler.ForceInline.DeviceFunctions': ('__device__ __forceinline__ float helper(float a, float b) { return a*b; }', 'Apply to current helper functions, not an unused added helper'),
        'Compiler.RestrictPointer': ('__global__ void gemm(..., const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) { ... }', 'Caller guarantees non-aliasing for the relevant accessed regions'),
        'Compiler.TemplateSpecialization.ShapeStatic': ('template<int M, int N, int K> __global__ void specialized_gemm(...) { ... }\n// Dispatch only the matching runtime shape.', 'Target shape known; retain dispatch guard'),
        'Compiler.Ptxas.MaxRegisterCount': ('nvcc --maxrregcount=REG_LIMIT ...\n// Read ptxas registers, spills and measured latency before accepting.', 'REG_LIMIT comes from bounded tuning, not an assumed optimal value'),
        'Compiler.LaunchBounds.MaxThreads': ('__global__ __launch_bounds__(THREADS) void gemm(...) { ... }', 'Actual CTA threads <= THREADS'),
        'Compiler.LaunchBounds.MinBlocksPerSM': ('__global__ __launch_bounds__(THREADS, MIN_BLOCKS) void gemm(...) { ... }', 'MIN_BLOCKS is a compiler hint; resource usage may prevent target residency'),
    }
    if s in exact:
        code, condition = exact[s]
    elif s.startswith('Memory.SharedMemory.Carveout.'):
        value = s.rsplit('.',1)[1]
        code = f'cudaError_t status = cudaFuncSetAttribute(kernel_instance, cudaFuncAttributePreferredSharedMemoryCarveout, {value});\n// Check status; this preference does not change the requested per-block allocation.'
        condition = 'Architecture supports the requested shared/L1 preference; preserve dynamic opt-in settings'
    elif s in ('Scheduling.WaveQuantization.SMResidentBlocks', 'Tuning.WarpOccupancyBalance',
               'Compiler.ResourceFeedback.PtxasOccupancySweep'):
        code = '''for setting in bounded_resource_candidates:
    compile_with(setting, ptxas_verbose=True)
    collect(registers, spills, static_shared, dynamic_shared, resident_blocks)
    run_correctness_then_measure_latency()
retain_fastest_valid_setting_without_changing_locked_tile_or_strategy()
# Wave analysis uses ceil(cta_count / (sm_count * resident_blocks)).'''
        condition = 'Use actual device/compiler occupancy and spills, not a guarantee derived from launch-bound syntax'
    else:
        raise ValueError('No curated compiler reference for '+s)
    return spec(code, [condition], ['Keep benchmark precision policy unchanged unless explicitly allowed', 'Compiler options must reach the real build command', 'Compare complete kernel latency and keep valid fallback'], 'build_or_tuning_recipe')


def cpu(s):
    if s.startswith('CPU.Tiling.'):
        group = s.split('.')[2]
        names = {'L2Block':('MC','NC','KC'), 'L1Block':('M1','N1','K1'), 'RegisterBlock':('MR','NR')}[group]
        values = s.rsplit('.',1)[1].split('x')
        return spec('\n'.join(f'enum {{ {n} = {v} }};' for n,v in zip(names,values)),
                    ['Ordinary CPU C, row-major FP32', 'Cache tile sizes and register tile sizes are different nesting levels'],
                    ['Tile loops use clipped end bounds', 'L1/L2 capacity estimates include all simultaneously live panels', 'Regenerate compatible packing and microkernel shapes'], 'deterministic_parameter_example')
    if s == 'CPU.Memory.NoPack':
        return spec('float a = A[(size_t)(i0+i)*lda + k0+k];\nfloat b = B[(size_t)(k0+k)*ldb + j0+j];',
                    ['Use original leading dimensions'], ['Do not keep packed-panel indexing after selecting NoPack', 'Preserve vector/loop strategy or report incompatibility'])
    if s.startswith(('CPU.Memory.Pack', 'CPU.Packing.')) and 'PackBuffer' not in s:
        operands = 'AB' if 'PackAB' in s else ('A' if 'PackA' in s else 'B')
        micro = s.startswith('CPU.Packing.')
        aindex = 'k*MR+i' if micro else 'i*kc+k'
        bindex = 'k*NR+j' if micro else 'k*nc+j'
        imax, jmax = ('MR','NR') if micro else ('mc','nc')
        lines = []
        if 'A' in operands:
            lines.append(f'for (int k=0; k<kc; ++k) for (int i=0; i<{imax}; ++i)\n  Ap[{aindex}] = (i0+i<M) ? A[(size_t)(i0+i)*lda+k0+k] : 0.0f;')
        if 'B' in operands:
            lines.append(f'for (int k=0; k<kc; ++k) for (int j=0; j<{jmax}; ++j)\n  Bp[{bindex}] = (j0+j<N) ? B[(size_t)(k0+k)*ldb+j0+j] : 0.0f;')
        return spec('\n'.join(lines), ['k0+kc <= K; edge panel lanes zero-filled', 'Allocate padded capacity with checked size_t arithmetic'],
                    ['Microkernel uses exactly this panel stride/order', 'Macro-panels contain explicit offsets for each MR/NR micropanel', 'No uninitialized padded values', 'Packed memory lifecycle and thread ownership valid'],
                    notes=['MRxKC names a logical shape; this reference uses k-major interleaved MR values. A row-major MRxKC consumer requires different indexing.'])
    if 'PackBuffer.' in s:
        tile = 'TilePrivate' in s
        code = '''#pragma omp parallel
{
  float *scratch = allocate_checked_aligned_scratch(panel_bytes);
  // Handle allocation failure before any thread enters a barrier-dependent path.
  #pragma omp for schedule(static)
  for (int tile=0; tile<tile_count; ++tile) {
    PACK_AND_COMPUTE_WITH_PRIVATE_SCRATCH(tile, scratch);
  }
  release_matching_allocator(scratch);
}'''
        if tile:
            code = '''#pragma omp parallel for schedule(static)
for (int tile=0; tile<tile_count; ++tile) {
  float *tile_scratch = allocate_checked_aligned_scratch(panel_bytes);
  if (tile_scratch == NULL) RECORD_TILE_ALLOCATION_FAILURE(tile);
  else {
    PACK_AND_COMPUTE_WITH_TILE_PRIVATE_SCRATCH(tile, tile_scratch);
    release_matching_allocator(tile_scratch);
  }
}
// Do not accept output when any tile allocation failed.'''
        return spec(code, ['Thread-private scratch never escapes its owning thread',
         'Tile-private policy requires a distinct slot/lifetime per concurrently live tile' if tile else 'Reuse thread scratch only after the preceding microkernel finishes'],
         ['No shared scratch writes by multiple threads', 'Check allocation failures and avoid barrier deadlocks', 'Pair _aligned_malloc/_aligned_free or aligned_alloc/free correctly'], 'algorithm_sketch_with_explicit_placeholders',
         ['The tile-private example allocates per tile; a preallocated tile-indexed arena can avoid allocation overhead without sharing live slots.'] if tile else [])
    if s.startswith('CPU.Memory.Prefetch.'):
        if s.endswith('Disabled'):
            return spec('// Remove explicit software prefetch instructions; retain required ordinary loads.',
                        ['Do not remove the actual operand loads'], ['No explicit prefetch left in generated hot loop'], 'code_removal_recipe')
        operands = 'AB' if s.endswith('ABPanel') else 'B'
        return spec('\n'.join(f'if (k + DISTANCE < kc) PREFETCH_READ(&{t}p[(size_t)(k+DISTANCE)*STRIDE_{t}]);' for t in operands),
                    ['Prefetch target remains inside the allocated panel', 'PREFETCH_READ maps to a supported compiler intrinsic, DISTANCE is tuned'],
                    ['No out-of-allocation pointer arithmetic even for nonfaulting prefetch', 'Benchmark prefetch overhead', 'Do not replace required loads'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith('CPU.MicroKernel.') or s.startswith('CPU.Vectorization.'):
        if any(x in s for x in ('Scalar.', 'ScalarPortable', 'CompilerAutoSIMD', 'PragmaSIMD')):
            pragma = '#pragma omp simd\n' if 'PragmaSIMD' in s else ''
            return spec('''for (int k=0; k<kc; ++k)
  for (int i=0; i<MR; ++i) {
    const float a = Ap[k*MR+i];
    '''+pragma+'''for (int j=0; j<NR; ++j) acc[i][j] += a * Bp[k*NR+j];
  }''', ['Packed k-major MR/NR panels; adapt if NoPack selected', 'Accumulator initialized once for this microkernel call', 'Scalar.4x4 uses MR=4, NR=4'],
                        ['Store only valid edge outputs', 'PragmaSIMD needs supported compiler flags; no reduction over K is parallelized here', 'CompilerAutoSIMD must be checked in compiler vectorization reports', 'ScalarPortable uses no ISA-specific intrinsics'])
        avx512 = 'AVX512' in s
        lanes,typ,prefix = (16,'__m512','_mm512') if avx512 else (8,'__m256','_mm256')
        dims = re.search(r'\.(\d+)x(\d+)$',s)
        mr,nr = (dims.group(1),dims.group(2)) if dims else ('MR','NR')
        return spec(f'''{typ} acc[{mr}][{nr}/{lanes}];
for (int i=0;i<{mr};++i) for (int q=0;q<{nr}/{lanes};++q) acc[i][q]={prefix}_setzero_ps();
for (int k=0;k<kc;++k) {{
  for (int q=0;q<{nr}/{lanes};++q) {{
    {typ} b={prefix}_loadu_ps(Bp+k*{nr}+q*{lanes});
    for (int i=0;i<{mr};++i) {{
      {typ} a={prefix}_set1_ps(Ap[k*{mr}+i]);
      acc[i][q]={prefix}_fmadd_ps(a,b,acc[i][q]);
    }}
  }}
}}
// Apply alpha/beta and store only owned valid outputs, with scalar/masked tails.''',
                    [f'Host supports {"AVX512F" if avx512 else "AVX2"} and FMA; OS enables the required vector state', f'NR divisible by {lanes}; packed A/B use MR/NR k-major strides', 'Compile intrinsics with the matching ISA flags and immintrin.h'],
                    ['Do not silently change selected register block to match an incompatible microkernel', 'No ISA instruction on unsupported CPUs', 'Edge buffers or masked stores prevent overruns', 'Inspect spills for large register tiles'])
    if s in ('CPU.LoopOrder.IJK', 'CPU.LoopOrder.IKJ'):
        if s.endswith('IJK'):
            code = 'for (int i=0;i<M;++i) for (int j=0;j<N;++j) {\n  float sum=0;\n  for (int k=0;k<K;++k) sum += A[(size_t)i*K+k]*B[(size_t)k*N+j];\n  C[(size_t)i*N+j]=alpha*sum+(beta==0?0:beta*C[(size_t)i*N+j]);\n}'
        else:
            code = 'for (size_t p=0;p<(size_t)M*N;++p) C[p]=(beta==0?0:beta*C[p]);\nfor (int i=0;i<M;++i) for (int k=0;k<K;++k) {\n  float a=alpha*A[(size_t)i*K+k];\n  for (int j=0;j<N;++j) C[(size_t)i*N+j] += a*B[(size_t)k*N+j];\n}'
        return spec(code, ['Row-major A/B/C, no packing in this reference'], ['Alpha/beta exactly once', 'Reordering may change FP rounding; test chosen tolerance', 'Adapt selected packing/microkernel rather than discard it'])
    if s in ('CPU.LoopOrder.TiledIJK','CPU.MacroKernel.DirectTiled.Driver'):
        return spec('''for (int ic=0;ic<M;ic+=MC) for (int jc=0;jc<N;jc+=NC)
  for (int pc=0;pc<K;pc+=KC) {
    int mc=min(MC,M-ic), nc=min(NC,N-jc), kc=min(KC,K-pc);
    DIRECT_TILE(ic,jc,pc,mc,nc,kc,pc==0?beta:1.0f);
  }''', ['DIRECT_TILE retains caller leading dimensions and handles edges'], ['Beta only on the first K panel', 'Accumulate all K panels', 'Materialize the driver without library calls'], 'algorithm_sketch_with_explicit_placeholders')
    if s in ('CPU.LoopOrder.PackedPanelMajor','CPU.LoopOrder.OpenBLASPanelMajor','CPU.MacroKernel.OpenBLASStyle.PanelDriver'):
        return spec('''for (int jc=0;jc<N;jc+=NC) for (int pc=0;pc<K;pc+=KC) {
  PACK_B_PANEL(pc,jc,min(KC,K-pc),min(NC,N-jc));
  for (int ic=0;ic<M;ic+=MC) {
    PACK_A_PANEL(ic,pc,min(MC,M-ic),min(KC,K-pc));
    for (int jr=0;jr<min(NC,N-jc);jr+=NR)
      for (int ir=0;ir<min(MC,M-ic);ir+=MR)
        MICROKERNEL(ir,jr,pc==0?beta:1.0f);
  }
}''', ['Concrete panel offsets must match the selected packing strategy'],
                    ['B packed once per jc/pc and reused over ic', 'Beta once over the entire reduction', 'Thread-private A and safely published B panels', 'This is an algorithm reference, not a call to OpenBLAS'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith('CPU.KLoop.'):
        if s.endswith('NoUnroll'):
            code = 'for (int k=0;k<kc;++k) ACCUMULATE_K(k);\n// Disable automatic unrolling using a supported compiler option if required.'
        else:
            u=s.rsplit('Unroll',1)[1]
            code = f'int k=0;\nfor (;k+{u}<=kc;k+={u}) for (int u=0;u<{u};++u) ACCUMULATE_K(k+u);\nfor (;k<kc;++k) ACCUMULATE_K(k);'
        return spec(code, ['ACCUMULATE_K uses current packed/direct microkernel operands'], ['Remainder preserved', 'Inspect generated code rather than assume loop syntax forces unrolling'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith(('CPU.Parallel.', 'CPU.Threading.')):
        if s.endswith('Serial'):
            code = 'for (int tile=0;tile<tile_count;++tile) COMPUTE_TILE(tile);\n// No OpenMP parallel region or internally threaded helper.'
        elif s.endswith('RowBlock'):
            code = '#pragma omp parallel for schedule(static)\nfor (int ic=0;ic<M;ic+=MC) COMPUTE_ALL_COLUMNS_AND_K_FOR_ROW_BLOCK(ic);'
        else:
            code = '#pragma omp parallel for collapse(2) schedule(static)\nfor (int ic=0;ic<M;ic+=MC)\n  for (int jc=0;jc<N;jc+=NC) COMPUTE_ALL_K_FOR_OUTPUT_TILE(ic,jc);'
        return spec(code, ['Compiler supports selected OpenMP syntax; current thread count policy is honored'],
                    ['Disjoint C ownership, not parallel K accumulation without reduction', 'Private accumulators and mutable packing buffers', 'No nested oversubscription'], 'algorithm_sketch_with_explicit_placeholders')
    if s.startswith('CPU.Compiler.'):
        flags = {'PortableO2':'cc -O2 -std=c11', 'NativeO3':'cc -O3 -march=native -std=c11',
                 'NativeO3OpenMP':'cc -O3 -march=native -fopenmp -std=c11',
                 'MSVC.AVX2OpenMP':'cl /O2 /arch:AVX2 /openmp',
                 'NativeO3FastMathOpenMP':'cc -O3 -march=native -ffast-math -fopenmp -std=c11'}[s.removeprefix('CPU.Compiler.')]
        return spec(flags+' main.c cpu_kernel.c ...', ['Flags are compiler/platform specific', 'Fast-math requires explicit numerical tolerance permission; native builds are machine-specific'],
                    ['Options reach the actual compile and link commands', 'CPU C only, no nvcc/CUDA dependency', 'Dispatch or reject ISA mismatch'], 'build_or_tuning_recipe')
    if s.startswith('CPU.Epilogue.'):
        value = 'alpha * sum' if s.endswith('BetaZeroFastPath') else 'alpha * sum + (beta == 0 ? 0 : beta * C[idx])'
        return spec('C[idx] = '+value+';', ['Valid output idx; beta==0 dispatch is required for BetaZeroFastPath'], ['No reading uninitialized C when beta==0', 'Apply beta once across K panels'])
    if s == 'CPU.TailKernel.FullTileFastPathScalarCleanup':
        return spec('''for (int i=0;i+MR<=M;i+=MR) for (int j=0;j+NR<=N;j+=NR) FULL_MICROKERNEL(i,j);
for (int i=0;i<M;++i) for (int j=0;j<N;++j)
  if (i>=M/MR*MR || j>=N/NR*NR) SCALAR_FULL_K_GEMM_OUTPUT(i,j);''',
                    ['Full microkernel covers complete K including its unroll tail'],
                    ['Main rectangle and cleanup are disjoint/exhaustive', 'Do not process corner twice', 'No out-of-range vector stores'], 'algorithm_sketch_with_explicit_placeholders')
    raise ValueError('No curated CPU reference for '+s)


def build():
    base = json.loads((ROOT/'strategy_examples.json').read_text(encoding='utf-8'))
    generated = {'schema_version':1, 'examples':{}, 'strategy_bindings':{}}
    libraries = {}
    counts = {}
    for filename, backend in (('strategy_library.json','gpu'), ('cpu_strategy_library.json','cpu')):
        library = json.loads((ROOT/filename).read_text(encoding='utf-8'))
        entries = list(nodes(library))
        counts[backend] = len(entries)
        for entry in entries:
            sid = entry['strategy_id']
            if sid in base['strategy_bindings']:
                entry['implementation_example_ids'] = base['strategy_bindings'][sid]
                continue
            detail = cpu(sid) if backend=='cpu' else gpu(sid)
            eid = 'strategy_reference.'+sid
            example = dict(example_id=eid, strategy_id=sid, backend=backend,
                validation_status='design_reference_not_independently_compiled_or_benchmarked',
                regions=entry.get('modifies_regions', []), **detail)
            if not example['regions']:
                example['regions'] = ['CURRENT_STRATEGY_OWNED_REGIONS']
            example['adaptation_rules'] += [
                'Current strategy pre/postconditions, IR and parent code take precedence over this reference.',
                'Upper-case helper calls, ellipses and symbolic configuration values are explanatory placeholders: implement or resolve them, never paste undefined calls.',
                'Reference shape/layout assumptions are not permission to change locked strategies. Report incompatible contracts instead.',
                'All numerical, safety, resource and strategy-realization verification remains mandatory for a complete kernel.']
            generated['examples'][eid] = example
            generated['strategy_bindings'][sid] = [eid]
            entry['implementation_example_ids'] = [eid]
        libraries[filename] = library
    ids = {e['strategy_id'] for lib in libraries.values() for e in nodes(lib)}
    if set(base['strategy_bindings']) - ids:
        raise ValueError('Base catalog references strategies absent from the libraries')
    covered = set(base['strategy_bindings']) | set(generated['strategy_bindings'])
    if covered != ids:
        raise ValueError('Incomplete example coverage')
    generated['coverage'] = {**counts, 'total':len(ids), 'missing':[]}
    return libraries, generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    libraries, generated = build()
    outputs = {**libraries, 'strategy_examples_full.json':generated}
    for filename, value in outputs.items():
        path = ROOT/filename
        if args.check:
            if not path.exists() or json.loads(path.read_text(encoding='utf-8')) != value:
                raise ValueError('Stale example metadata: '+str(path))
        else:
            path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(generated['coverage']))


if __name__ == '__main__':
    main()
