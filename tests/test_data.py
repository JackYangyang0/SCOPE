"""Input data: counts, determinism, split separation, frozen hashes."""

from __future__ import annotations

import unittest

import numpy as np

from SCOPE.data import matrix_cases as data

# Captured from the frozen generator.  If these move, the data changed; the
# correct response is a new GENERATOR_VERSION and a new ledger, never an update
# in place (see data/matrix_cases.py).
GOLDEN = {
    "shape": 16,
    "development": {
        "A": "5c798a5e3655db55ea6991c3b5297ca736afbf4410d78986fb278ca1969beff7",
        "B": "b5b47e43f62224dba2205018f4abfbfa9a4d28485a310062328685342f497dc8",
        "C0": "7dd9c45c7eee6c78140615cbc556f8b07c48f484f4f3f0f848f4f304b6f13fb3",
        "ledger": "4e7605c2a4ef46a805b921d3ff056177e25921cb40a60371aa35da447293b532",
    },
    "postselection_test": {
        "ledger": "b448a16965856a26bba7272ae835e8552ea8021e2df2d35cd59682d866559987",
    },
}


class DataInventoryTest(unittest.TestCase):
    def test_config_count_and_families(self):
        self.assertEqual(len(data.CONFIGS), 32)
        self.assertEqual(len(data.CONFIG_IDS), 32)
        self.assertEqual(sum(data.FAMILY_COUNTS.values()), 32)
        self.assertEqual(data.FAMILY_COUNTS["uniform_small"], 10)
        self.assertEqual(data.FAMILY_COUNTS["uniform_wide"], 10)
        self.assertEqual(data.FAMILY_COUNTS["normal"], 3)
        self.assertEqual(data.FAMILY_COUNTS["sparse"], 3)
        self.assertEqual(data.FAMILY_COUNTS["original_deterministic"], 1)
        self.assertTrue(data.CONFIGS["original_deterministic"]["compat_only"])

    def test_check_inventory(self):
        manifest = data.build_manifest([8], split="development")
        counts = manifest["per_shape_counts"]["8"]
        self.assertEqual(counts["main"], 64)          # 32 configs x 2 C0
        self.assertEqual(counts["coefficients"], 128)  # 2 pairs x 32 x 2
        self.assertEqual(counts["total"], 192)
        self.assertEqual(manifest["entry_count"], 192)

    def test_two_c0_variants_are_different(self):
        pattern = data.make_case("uniform_small.s00", 8, 8, 8, c0_variant="pattern")
        scalar = data.make_case("uniform_small.s00", 8, 8, 8, c0_variant="scalar")
        self.assertNotEqual(pattern["hashes"]["C0"], scalar["hashes"]["C0"])
        self.assertEqual(pattern["hashes"]["A"], scalar["hashes"]["A"])
        self.assertTrue(np.all(scalar["C0"] == np.float32(-123.25)))


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_bytes(self):
        first = data.make_case("uniform_wide.s07", 12, 12, 12, split="development")
        second = data.make_case("uniform_wide.s07", 12, 12, 12, split="development")
        self.assertEqual(first["hashes"], second["hashes"])

    def test_splits_do_not_share_bytes(self):
        development = data.make_case("uniform_small.s00", 12, 12, 12,
                                     split="development")
        generation = data.make_case("uniform_small.s00", 12, 12, 12,
                                    split="generation")
        test = data.make_case("uniform_small.s00", 12, 12, 12,
                              split="postselection_test")
        self.assertNotEqual(development["hashes"]["A"], generation["hashes"]["A"])
        self.assertNotEqual(generation["hashes"]["A"], test["hashes"]["A"])
        self.assertNotEqual(development["hashes"]["A"], test["hashes"]["A"])

    def test_seed_blocks_disjoint(self):
        data.assert_split_disjoint()
        random_configs = [config_id for config_id in data.CONFIG_IDS
                          if data.CONFIGS[config_id]["random"]]
        seeds = {data.case_seed(config_id, split)
                 for config_id in random_configs for split in data.SPLITS}
        self.assertEqual(len(seeds), len(random_configs) * len(data.SPLITS))
        block_start = min(data.SEED_BASE.values())
        self.assertTrue(all(seed >= block_start for seed in seeds))
        # Structural cases are split-independent by design, and their seeds must
        # stay outside every random block.
        for config_id, seed in data.STRUCTURAL_SEEDS.items():
            self.assertLess(seed, block_start, config_id)
            for split in data.SPLITS:
                self.assertEqual(data.case_seed(config_id, split), seed)

    def test_golden_hashes(self):
        case = data.make_case("uniform_small.s00", GOLDEN["shape"], GOLDEN["shape"],
                              GOLDEN["shape"], split="development")
        for name, expected in GOLDEN["development"].items():
            if name == "ledger":
                continue
            self.assertEqual(case["hashes"][name], expected, name)
        ledger = data.golden_hashes([GOLDEN["shape"]], split="development")
        self.assertEqual(ledger["ledger_hash"], GOLDEN["development"]["ledger"])
        post = data.golden_hashes([GOLDEN["shape"]], split="postselection_test")
        self.assertEqual(post["per_shape"][str(GOLDEN["shape"])],
                         GOLDEN["postselection_test"]["ledger"])


class StructuralReferenceTest(unittest.TestCase):
    """Each structural family must have the property its name claims."""

    def test_zero_is_zero(self):
        case = data.make_case("zero", 8, 8, 8)
        self.assertFalse(np.any(case["A"]))
        self.assertFalse(np.any(case["B"]))

    def test_identity_families(self):
        left = data.make_case("left_identity", 8, 8, 8)
        self.assertTrue(np.array_equal(np.eye(8, dtype=np.float32), left["A"]))
        right = data.make_case("right_identity", 8, 8, 8)
        self.assertTrue(np.array_equal(np.eye(8, dtype=np.float32), right["B"]))

    def test_impulse_has_one_nonzero_output(self):
        case = data.make_case("impulse", 8, 8, 8)
        product = case["A"].astype(np.float64) @ case["B"].astype(np.float64)
        self.assertEqual(product[0, 0], 2.0)
        self.assertEqual(float(np.abs(product).sum()), 2.0)

    def test_cancellation_exact_value_any_order(self):
        """Pairs cancel to a representable value, so order cannot matter."""
        case = data.make_case("cancellation", 8, 8, 8)
        pairs = 8 // 2
        expected = pairs * 2.0 ** -20
        A = case["A"].astype(np.float64)
        B = case["B"].astype(np.float64)
        self.assertAlmostEqual(float((A @ B)[0, 0]), expected, places=18)
        # Reversing the k order must give exactly the same value.
        reversed_product = float((A[:, ::-1] @ B[::-1, :])[0, 0])
        self.assertEqual(reversed_product, expected)

    def test_compat_case_matches_historical_bytes(self):
        case = data.make_case("original_deterministic", 6, 5, 4)
        A, B = case["A"], case["B"]
        self.assertEqual(A.shape, (6, 4))
        self.assertEqual(B.shape, (4, 5))
        ia = np.arange(24, dtype=np.int64)
        self.assertTrue(np.array_equal(A.reshape(-1), (ia // 13).astype(np.float32)))
        ib = np.arange(20, dtype=np.int64)
        self.assertTrue(np.array_equal(B.reshape(-1), (ib % 13).astype(np.float32)))


class LayoutContractTest(unittest.TestCase):
    def test_every_case_is_c_contiguous_float32_with_matching_shape(self):
        for config_id in ("uniform_small.s00", "normal.s01", "sparse.s02",
                          "cancellation", "left_identity",
                          "original_deterministic"):
            for M, N, K in ((7, 9, 5), (12, 12, 12)):
                case = data.make_case(config_id, M, N, K)
                self.assertEqual(case["A"].shape, (M, K), config_id)
                self.assertEqual(case["B"].shape, (K, N), config_id)
                self.assertEqual(case["C0"].shape, (M, N), config_id)
                for name in ("A", "B", "C0"):
                    array = case[name]
                    self.assertEqual(array.dtype, np.float32, name)
                    self.assertTrue(array.flags["C_CONTIGUOUS"], name)
                self.assertEqual(case["layout"], "row_major")
                self.assertEqual(case["dtype"], "float32")

    def test_case_key_binds_coefficients_and_variant(self):
        unit = data.make_case("uniform_small.s00", 8, 8, 8,
                              coefficients=data.MAIN_COEFFICIENTS)
        beta = data.make_case("uniform_small.s00", 8, 8, 8,
                              coefficients=data.COEFFICIENT_SETS[0])
        scalar = data.make_case("uniform_small.s00", 8, 8, 8, c0_variant="scalar")
        self.assertNotEqual(data.case_key(unit), data.case_key(beta))
        self.assertNotEqual(data.case_key(unit), data.case_key(scalar))


if __name__ == "__main__":
    unittest.main()
