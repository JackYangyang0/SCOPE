import tempfile
import unittest
from pathlib import Path

from SCOPE.specialization.shape_specialization import resolve_config, load_existing_seed


class ShapeSpecializationTests(unittest.TestCase):
    def test_current_and_existing_modes(self):
        config = {'shape_specialization': {'enabled': True, 'seed_source': 'current_top1'}}
        self.assertEqual(resolve_config(config)['seed_source'], 'current_top1')
        self.assertEqual(resolve_config(config, reuse_family=Path('seed'))['seed_source'], 'existing')
        self.assertFalse(resolve_config(config, 'off')['enabled'])
        with self.assertRaises(ValueError):
            resolve_config(config, 'only')
        self.assertEqual(config['shape_specialization']['seed_source'], 'current_top1')

    def test_existing_without_measurements_requires_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'cuda_kernel.cuh').write_text('kernel', encoding='utf-8')
            (folder / 'main.cpp').write_text('main', encoding='utf-8')
            seed = load_existing_seed(folder, {'problem': {'M': 512}, 'hardware': {'name': 'current'}})
            self.assertTrue(seed['requires_revalidation'])
            self.assertNotIn('accepted', seed)
            self.assertEqual(seed['source_snapshot']['cuda_kernel.cuh'], 'kernel')
            self.assertEqual(seed['verified_ir']['hardware']['name'], 'current')
