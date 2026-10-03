"""Exhaustive hierarchical enumeration within the declared strategy domains."""
from SCOPE.generate_ir.tiling_planner import (
    collect_strategy_ids, parse_strategy_tiles, BLOCK_TILE_RE, WARP_TILE_RE,
    THREAD_TILE_RE, first_nested_int,
)
from SCOPE.generate_ir.stage_controller import divisors


def enumerate_tiles(library, ir):
    ids = collect_strategy_ids(library)
    blocks = sorted(set(t for _, t in parse_strategy_tiles(ids, BLOCK_TILE_RE)))
    warps = sorted(set(t for _, t in parse_strategy_tiles(ids, WARP_TILE_RE)))
    threads = sorted(set(t for _, t in parse_strategy_tiles(ids, THREAD_TILE_RE)))
    warp_size = first_nested_int(ir, ['hardware.warp_size', 'hardware.gpu.warp_size'], 32)
    limit = first_nested_int(ir, ['hardware.max_threads_per_block', 'hardware.gpu.max_threads_per_block'], 1024)
    for bm, bn, bk in blocks:
        if min(bm, bn, bk) <= 0:
            continue
        for wm, wn in warps:
            if min(wm, wn) <= 0 or bm % wm or bn % wn:
                continue
            count = (bm // wm) * (bn // wn)
            if count * warp_size > limit:
                continue
            for tm, tn in threads:
                if min(tm, tn) <= 0:
                    continue
                for wmi in divisors(wm):
                    if wmi % tm:
                        continue
                    for wni in divisors(wn):
                        if wni % tn or (wmi // tm) * (wni // tn) != warp_size:
                            continue
                        yield dict(BM=bm, BN=bn, BK=bk, WM=wm, WN=wn,
                                   TM=tm, TN=tn, WMITER=wmi, WNITER=wni,
                                   warps_per_block=count, threads_per_block=count * warp_size)
