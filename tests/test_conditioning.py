import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from lib_krea2_edit.conditioning import (
    GroundedQwenEngine,
    forge_qwen_vision_attention_compat,
    install_grounded_setup,
)


class RecordingEngine:
    marker = "delegated"

    def __init__(self):
        self.calls = []

    def __call__(self, texts, images=None):
        self.calls.append((texts, images))
        return texts


class FakeModel:
    def __init__(self, engine):
        self.text_processing_engine_qwen = engine


class FakeProcessing:
    def __init__(self, engine):
        self.sd_model = FakeModel(engine)
        self.encoded = None

    def setup_conds(self):
        self.encoded = self.sd_model.text_processing_engine_qwen(
            ["positive", "negative"]
        )


class ConditioningTests(unittest.TestCase):
    def test_proxy_injects_image_and_delegates_attributes(self):
        engine = RecordingEngine()
        image = object()
        proxy = GroundedQwenEngine(engine, image)
        proxy(["prompt"])
        self.assertEqual(engine.calls, [(["prompt"], [image])])
        self.assertEqual(proxy.marker, "delegated")

    def test_setup_swap_is_scoped_and_idempotent(self):
        engine = RecordingEngine()
        image = object()
        processing = FakeProcessing(engine)
        install_grounded_setup(processing, image)
        install_grounded_setup(processing, object())
        with patch(
            "lib_krea2_edit.conditioning.forge_qwen_vision_attention_compat",
            return_value=nullcontext(),
        ):
            processing.setup_conds()
        self.assertIs(processing.sd_model.text_processing_engine_qwen, engine)
        self.assertEqual(engine.calls, [(["positive", "negative"], [image])])

    def test_qwen_vision_attention_adapter_returns_forge_backend_and_restores(self):
        original = object()
        selected_attention = object()
        qwen_module = SimpleNamespace(attention_function=original)
        with forge_qwen_vision_attention_compat(qwen_module, selected_attention):
            selected = qwen_module.attention_function(
                "cpu", mask=False, small_input=True
            )
            self.assertIs(selected, selected_attention)
        self.assertIs(qwen_module.attention_function, original)


if __name__ == "__main__":
    unittest.main()
