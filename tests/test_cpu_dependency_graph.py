from __future__ import annotations

import json
import unittest
from pathlib import Path

from SCOPE.app import ROOT
from SCOPE.generate_ir.strategy_index_filter import find_missing_requirements


class CpuDependencyGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = json.loads((ROOT / "data/graph/cpu_dependency_graph.json").read_text(encoding="utf-8"))
        cls.index = json.loads((ROOT / "data/lib/cpu_strategy_index.json").read_text(encoding="utf-8"))

    def missing(self, strategy_id, ir, applied):
        return find_missing_requirements(strategy_id, ir, set(applied), self.graph)

    def test_strategy_and_state_requirements_are_both_mandatory(self):
        strategy = "CPU.MacroKernel.OpenBLASStyle.PanelDriver"
        self.assertTrue(self.missing(strategy, {"cpu_memory": {"pack_b": False}}, {"CPU.Memory.PackBuffer.ThreadPrivate"}))
        self.assertFalse(self.missing(strategy, {"cpu_memory": {"pack_b": True}}, {"CPU.Memory.PackBuffer.ThreadPrivate"}))

    def test_microkernel_shape_must_match_register_tile(self):
        strategy = "CPU.MicroKernel.AVX2.FMA.4x8"
        self.assertTrue(self.missing(strategy, {"cpu_tiling": {"register_m": 4, "register_n": 4}}, set()))
        self.assertFalse(self.missing(strategy, {"cpu_tiling": {"register_m": 4, "register_n": 8}}, set()))

    def test_vectorization_must_match_selected_microkernel(self):
        strategy = "CPU.Vectorization.AVX2.FMA.Explicit"
        ir = {"cpu_microkernel": {"family": "avx2_fma"}}
        self.assertTrue(self.missing(strategy, ir, {"CPU.MicroKernel.Scalar.4x4"}))
        self.assertFalse(self.missing(strategy, ir, {"CPU.MicroKernel.AVX2.FMA.4x8"}))

    def test_prefetch_requires_materialized_packing(self):
        strategy = "CPU.Memory.Prefetch.ABPanel"
        applied = {"CPU.Packing.PackAB.MRxKC_KCxNR"}
        self.assertTrue(self.missing(strategy, {"cpu_memory": {"pack_a": True, "pack_b": False}}, applied))
        self.assertFalse(self.missing(strategy, {"cpu_memory": {"pack_a": True, "pack_b": True}}, applied))

    def test_graph_strategy_references_exist_in_cpu_index(self):
        known = {item["strategy_id"] for item in self.index["strategies"]}
        referenced = set(self.graph.get("requires", {})) | set(self.graph.get("fallback", {}))
        for section in ("requires", "fallback"):
            for entries in self.graph.get(section, {}).values():
                for entry in entries:
                    referenced.update(entry.get("strategies", []))
        self.assertEqual(set(), referenced - known)

    def test_cpu_index_and_library_are_bidirectionally_aligned(self):
        library = json.loads((ROOT / "data/lib/cpu_strategy_library.json").read_text(encoding="utf-8"))
        indexed = {item["strategy_id"] for item in self.index["strategies"]}
        implemented = {
            strategy["strategy_id"]
            for stage in library["stages"]
            for subphase in stage.get("ordered_subphases", [])
            for strategy in subphase.get("strategies", [])
        }
        self.assertEqual(indexed, implemented)

    def test_every_simd_microkernel_has_matching_register_tile(self):
        known = {item["strategy_id"] for item in self.index["strategies"]}
        for microkernel in (
            "CPU.MicroKernel.AVX2.FMA.4x8",
            "CPU.MicroKernel.AVX2.FMA.4x16",
            "CPU.MicroKernel.AVX2.FMA.6x16",
            "CPU.MicroKernel.AVX512.FMA.8x16",
        ):
            shape = microkernel.rsplit(".", 1)[-1]
            self.assertIn(f"CPU.Tiling.RegisterBlock.{shape}", known)


if __name__ == "__main__":
    unittest.main()
