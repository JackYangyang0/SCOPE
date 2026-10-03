import itertools
import unittest
from SCOPE.specialization.tile_enumeration import enumerate_tiles


class TileEnumerationTests(unittest.TestCase):
    def test_matches_brute_force_mapping_constraints(self):
        library = [{'strategy_id': s} for s in [
            'Tiling.BlockTile.64x64x16', 'Tiling.WarpTile.32x32',
            'Tiling.WarpTile.128x32', 'Tiling.ThreadTile.2x2', 'Tiling.ThreadTile.4x4']]
        tiles = list(enumerate_tiles(library, {}))
        expected = {(tm, tn, a, b) for tm, tn in [(2, 2), (4, 4)]
                    for a, b in itertools.product(range(1, 33), repeat=2)
                    if 32 % a == 0 and 32 % b == 0 and a % tm == 0 and b % tn == 0
                    and (a // tm) * (b // tn) == 32}
        self.assertEqual({(t['TM'], t['TN'], t['WMITER'], t['WNITER']) for t in tiles}, expected)
        self.assertEqual(len(tiles), len(expected))
        self.assertGreater(len(expected), 2)
        self.assertTrue(all(t['WM'] == 32 and t['threads_per_block'] == 128 for t in tiles))
        self.assertEqual(list(enumerate_tiles(library, {'hardware': {'max_threads_per_block': 64}})), [])


if __name__ == '__main__':
    unittest.main()
