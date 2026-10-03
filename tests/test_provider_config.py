import unittest
from unittest.mock import patch

from SCOPE.llm.provider_config import resolve_provider_config, PROVIDERS
from SCOPE.llm.openai_client import OpenAICompatibleClient


class ProviderConfigTests(unittest.TestCase):
    def config(self):
        return {'provider': 'cmecloud', **{
            name: {'model': name + '-model', 'base_url': 'https://example.invalid/' + name,
                   'api_key': name + '-test-key', 'timeout_seconds': 42}
            for name in PROVIDERS}}

    def test_all_providers_select_only_their_settings(self):
        for name in PROVIDERS:
            config = self.config()
            config['provider'] = name
            config['api_key'] = 'stale-root-key'
            with patch('SCOPE.llm.openai_client.OpenAI') as sdk:
                client = OpenAICompatibleClient(config)
                self.assertEqual(client._config['model'], name + '-model')
                self.assertEqual(sdk.call_args.kwargs['api_key'], name + '-test-key')
                self.assertEqual(sdk.call_args.kwargs['base_url'], 'https://example.invalid/' + name)
                self.assertEqual(client.timeout_seconds, 42)
                self.assertIsNone(client._thinking_mode)

    def test_legacy_is_unchanged_and_not_mutated(self):
        config = {'provider': 'custom', 'model': 'old', 'api_key': 'test'}
        resolved = resolve_provider_config(config)
        self.assertEqual(config, resolved)
        resolved['model'] = 'changed'
        self.assertEqual(config['model'], 'old')

    def test_invalid_selection(self):
        for value in ('wrong', None):
            config = self.config()
            config['provider'] = value
            with self.assertRaises(ValueError):
                resolve_provider_config(config)

    def test_inactive_empty_profiles_do_not_break_active_profile(self):
        config = self.config()
        config['qwen'] = {}
        self.assertEqual(resolve_provider_config(config)['model'], 'cmecloud-model')
        config['provider'] = 'qwen'
        with self.assertRaisesRegex(ValueError, 'llm.qwen.model'):
            resolve_provider_config(config)

    def test_environment_key_and_explicit_thinking(self):
        config = self.config()
        config['cmecloud'].pop('api_key')
        config['cmecloud'].update(api_key_env='SCOPE_TEST_KEY', thinking='disabled', codegen_timeout_seconds=90)
        with patch.dict('os.environ', {'SCOPE_TEST_KEY': 'test-secret'}), patch('SCOPE.llm.openai_client.OpenAI') as sdk:
            client = OpenAICompatibleClient(config)
            self.assertEqual(sdk.call_args.kwargs['api_key'], 'test-secret')
            self.assertEqual(client._thinking_mode, 'disabled')
            self.assertEqual(client.codegen_timeout_seconds, 90)


if __name__ == '__main__':
    unittest.main()
