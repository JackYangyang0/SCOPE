from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from SCOPE.llm.openai_client import OpenAICompatibleClient, resolve_api_key, require_final_content


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = '{"selected": true}'
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAI:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    def with_options(self, **kwargs):
        return self


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_environment_api_key_and_cme_compatible_json_request(self):
        fake = FakeOpenAI()
        config = {
            "model": "glm-5.2",
            "api_key_env": "SCOPE_TEST_CME_KEY",
            "base_url": "https://example.invalid/v1",
            "temperature": 0.0,
            "max_tokens": 4096,
            "selection_max_tokens": 1024,
            "use_json_response_format": False,
        }
        with patch.dict(os.environ, {"SCOPE_TEST_CME_KEY": "secret"}), patch(
            "SCOPE.llm.openai_client.OpenAI", return_value=fake
        ) as constructor:
            client = OpenAICompatibleClient(config)
            result = client.complete_selection_json([{"role": "user", "content": "json"}])

        self.assertTrue(result["selected"])
        self.assertEqual(constructor.call_args.kwargs["api_key"], "secret")
        request = fake.completions.calls[0]
        self.assertEqual(request["model"], "glm-5.2")
        self.assertEqual(request["max_tokens"], 1024)
        self.assertNotIn("response_format", request)
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})

    def test_json_response_format_remains_available_for_other_providers(self):
        fake = FakeOpenAI()
        with patch("SCOPE.llm.openai_client.OpenAI", return_value=fake):
            client = OpenAICompatibleClient({
                "model": "model",
                "api_key": "secret",
                "base_url": "https://example.invalid/v1",
                "temperature": 0.1,
                "use_json_response_format": True,
            })
            client.complete_json([{"role": "user", "content": "json"}])

        self.assertEqual(
            fake.completions.calls[0]["response_format"],
            {"type": "json_object"},
        )

    def test_shell_style_environment_placeholder_is_supported(self):
        with patch.dict(os.environ, {"SCOPE_TEST_PLACEHOLDER": "value"}):
            self.assertEqual(resolve_api_key({"api_key": "${SCOPE_TEST_PLACEHOLDER}"}), "value")

    def test_selection_effort_and_budget_do_not_change_codegen(self):
        fake = FakeOpenAI()
        with patch('SCOPE.llm.openai_client.OpenAI', return_value=fake):
            client = OpenAICompatibleClient({'model': 'glm-5.3', 'api_key': 'secret',
                'thinking': 'enabled', 'selection_reasoning_effort': 'low',
                'selection_max_tokens': 8192, 'codegen_max_tokens': 4096})
            client.complete_selection_json([])
            client.complete_tile_plan_text([])
            client.complete_codegen_json([])
        for call in fake.completions.calls[:2]:
            self.assertEqual(call['max_tokens'], 8192)
            self.assertEqual(call['extra_body'], {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'})
        self.assertEqual(fake.completions.calls[2]['max_tokens'], 4096)
        self.assertNotIn('reasoning_effort', fake.completions.calls[2]['extra_body'])

    def test_truncated_output_never_becomes_empty_selection(self):
        for content in ('', '{"selections": []}'):
            response = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason='length', message=SimpleNamespace(content=content))])
            with self.assertRaisesRegex(RuntimeError, 'output truncated'):
                require_final_content(response)

    def test_empty_final_content_is_explicit_error(self):
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content='  '))])
        with self.assertRaisesRegex(RuntimeError, 'empty final content'):
            require_final_content(response)


if __name__ == "__main__":
    unittest.main()
