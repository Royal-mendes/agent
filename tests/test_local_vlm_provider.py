import os
import unittest

from agent.schemas import AgentConfig
from agent.vlm_provider import build_vlm_provider


class FakeOpenAIClient:
    class _Message:
        content = '{"selected_skill":"GEOMETRIC_EXPLORE","skill_args":{},"confidence":0.6}'

    class _Choice:
        message = None

    class _Response:
        choices = []

    def __init__(self):
        self.last_kwargs = None
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        choice = self._Choice()
        choice.message = self._Message()
        response = self._Response()
        response.choices = [choice]
        return response


class LocalVLMProviderTests(unittest.TestCase):
    def test_local_provider_defaults_to_qwen_vllm_endpoint(self):
        old_base = os.environ.pop("LOCAL_VLM_BASE_URL", None)
        old_model = os.environ.pop("LOCAL_VLM_MODEL", None)
        old_tokens = os.environ.pop("LOCAL_VLM_MAX_TOKENS", None)
        try:
            provider = build_vlm_provider(AgentConfig(vlm_provider="local"))
            self.assertEqual(provider.model, "qwen2.5-vl-7b-instruct")
            self.assertEqual(provider.client.base_url.host, "127.0.0.1")
            self.assertEqual(provider.client.base_url.port, 18000)
            self.assertEqual(provider.max_tokens, 512)
            self.assertEqual(provider.extra_body["repetition_penalty"], 1.05)
        finally:
            if old_base is not None:
                os.environ["LOCAL_VLM_BASE_URL"] = old_base
            if old_model is not None:
                os.environ["LOCAL_VLM_MODEL"] = old_model
            if old_tokens is not None:
                os.environ["LOCAL_VLM_MAX_TOKENS"] = old_tokens

    def test_local_provider_generation_passes_qwen_sampling_args(self):
        fake_client = FakeOpenAIClient()
        provider = build_vlm_provider(
            AgentConfig(
                vlm_provider="local",
                vlm_model="qwen-local",
                vlm_base_url="http://127.0.0.1:18000/v1",
                vlm_api_key="local",
            )
        )
        provider.client = fake_client
        text = provider.generate("system", "user")
        self.assertIn("GEOMETRIC_EXPLORE", text)
        self.assertEqual(fake_client.last_kwargs["model"], "qwen-local")
        self.assertEqual(fake_client.last_kwargs["max_tokens"], 512)
        self.assertEqual(fake_client.last_kwargs["extra_body"]["repetition_penalty"], 1.05)


if __name__ == "__main__":
    unittest.main()
