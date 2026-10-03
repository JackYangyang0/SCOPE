import unittest

from SCOPE.app import select_stage_survivors, take_frontier_batch_with_lineage_floor


def state(path, gflops=0, shared=0):
    return {"path_code": path, "current_ir": {
        "performance": {"gflops": gflops},
        "resource": {"shared_memory": {"total_bytes": shared}},
    }}


def architecture_state(path, score, shared=0):
    result = state(path, shared=shared)
    result["current_ir"]["resource"]["tiling_candidate"] = {"architecture_score": score}
    return result


class StageSurvivorTests(unittest.TestCase):
    def test_lineage_floor_uses_one_global_top_three_budget(self):
        states = []
        for root, block, scores in (
            (1, "64x128x16", (100, 99, 98)),
            (2, "128x64x16", (80, 79, 78)),
            (3, "64x64x16", (60, 59, 58)),
        ):
            for child, score in enumerate(scores, start=1):
                candidate = architecture_state([root, child], score)
                candidate["path"] = [f"Tiling.BlockTile.{block}", f"child.{child}"]
                states.append(candidate)

        kept, removed = take_frontier_batch_with_lineage_floor(states, 3)

        self.assertEqual(len(kept), 3)
        self.assertEqual(len(removed), 6)
        self.assertEqual(
            {item["path"][0] for item in kept},
            {
                "Tiling.BlockTile.64x128x16",
                "Tiling.BlockTile.128x64x16",
                "Tiling.BlockTile.64x64x16",
            },
        )

    def test_global_top_three_after_full_expansion(self):
        states = [state([root, child]) for root in range(1, 4) for child in range(1, 4)]
        kept, removed = select_stage_survivors(states, 3)
        self.assertEqual(len(kept), 3)
        self.assertEqual(len(removed), 6)
        self.assertEqual([x["path_code"] for x in kept], [[1, 1], [1, 2], [2, 1]])

    def test_lineage_floor_keeps_best_family_per_tiling_root(self):
        states = []
        for root, block in enumerate(("64x128x16", "128x64x16", "64x64x16"), start=1):
            for child, layout in enumerate((
                "Layout.SharedMemory.PaddingAB.Plus1",
                "Layout.SharedMemory.TransposeB",
                "Layout.SharedMemory.SkewA.Xor",
            ), start=1):
                candidate = architecture_state([root, child], 100 - child)
                candidate["path"] = [f"Tiling.BlockTile.{block}", layout]
                states.append(candidate)
        kept, fallback = take_frontier_batch_with_lineage_floor(states, 3)
        self.assertEqual(
            {item["path"][1] for item in kept},
            {"Layout.SharedMemory.PaddingAB.Plus1"},
        )
        self.assertIn("Layout.SharedMemory.TransposeB", {item["path"][1] for item in fallback})
        self.assertIn("Layout.SharedMemory.SkewA.Xor", {item["path"][1] for item in fallback})

    def test_measured_performance_selects_stage_kernel(self):
        best = state([1], gflops=1)
        fast = state([3], gflops=100000)
        self.assertEqual(select_stage_survivors([fast, best], 1)[0], [fast])

    def test_resource_estimate_breaks_equal_rank_ties(self):
        low = state([1, 2], shared=1024)
        high = state([2, 1], shared=32768)
        self.assertEqual(select_stage_survivors([high, low], 1)[0], [low])

    def test_fewer_candidates_are_kept(self):
        states = [state([1]), state([2])]
        self.assertEqual(select_stage_survivors(states, 3), (states, []))

    def test_architecture_score_drives_stage_pruning_when_available(self):
        low = architecture_state([1], 20.0, shared=1024)
        high = architecture_state([3], 80.0, shared=32768)
        self.assertEqual(select_stage_survivors([low, high], 1)[0], [high])

    def test_mmr_preserves_structurally_different_candidate(self):
        states = [architecture_state([1], 100), architecture_state([2], 99), architecture_state([3], 98)]
        states[0]["current_ir"]["tiling"] = {"block_m": 64, "block_n": 64}
        states[1]["current_ir"]["tiling"] = {"block_m": 64, "block_n": 64}
        states[2]["current_ir"]["tiling"] = {"block_m": 128, "block_n": 32}
        kept, _ = select_stage_survivors(states, 2)
        self.assertIn(states[2], kept)


if __name__ == "__main__":
    unittest.main()
