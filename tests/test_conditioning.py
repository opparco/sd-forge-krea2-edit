import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lib_krea2_edit.conditioning import (
    GroundedQwenEngine,
    expand_grounded_multipliers,
    forge_qwen_vision_attention_compat,
    grounded_token_multiplier_compat,
    install_grounded_setup,
)


class RecordingEngine:
    marker = "delegated"

    def __init__(self):
        self.calls = []

    def __call__(self, texts, images=None):
        self.calls.append((texts, images))
        return texts


class RecordingEmphasis:
    def after_transformers(self):
        self.z = self.z * self.multipliers.unsqueeze(-1)


class FakeGroundedEngine:
    def __init__(self):
        self.emphasis = RecordingEmphasis()
        self.process_tokens = self.original_process_tokens
        self.text_encoder = self.encode
        self.encoder_call = None

    @staticmethod
    def original_process_tokens(batch_tokens, batch_multipliers):
        return "original"

    @staticmethod
    def process_embeds(batch_tokens):
        embeds = torch.ones(1, 7, 2)
        mask = torch.ones(1, 7, dtype=torch.long)
        count = [7]
        info = [{"type": "image", "index": 1, "size": 4}]
        return embeds, mask, count, info

    def encode(self, *args, **kwargs):
        self.encoder_call = kwargs
        return None, kwargs["embeds"]


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
    def test_image_placeholder_multiplier_expands_to_vision_token_count(self):
        tokens = [151644, {"type": "image"}, 123, 151645]
        multipliers = [1.0, 0.75, 1.2, 1.0]
        info = [{"type": "image", "index": 1, "size": 4}]
        expanded = expand_grounded_multipliers(tokens, multipliers, info)
        self.assertEqual(expanded, [1.0, 0.75, 0.75, 0.75, 0.75, 1.2, 1.0])

    def test_observed_qwen_expansion_matches_embedding_length(self):
        tokens = [0] * 138 + [{"type": "image"}]
        multipliers = [1.0] * 139
        info = [{"type": "image", "index": 138, "size": 456}]
        expanded = expand_grounded_multipliers(tokens, multipliers, info)
        self.assertEqual(len(expanded), 594)

    def test_grounded_process_tokens_uses_expanded_weights_and_restores(self):
        engine = FakeGroundedEngine()
        original = engine.process_tokens
        tokens = [[1, {"type": "image"}, 2, 3]]
        multipliers = [[1.0, 0.5, 2.0, 1.0]]
        with grounded_token_multiplier_compat(engine, tap_layers=[2, 5]):
            output = engine.process_tokens(tokens, multipliers)
        self.assertEqual(tuple(output.shape), (1, 7, 2))
        self.assertEqual(output[0, :, 0].tolist(), [1.0, 0.5, 0.5, 0.5, 0.5, 2.0, 1.0])
        self.assertEqual(engine.encoder_call["intermediate_output"], [2, 5])
        self.assertIs(engine.process_tokens, original)

    def test_proxy_injects_image_and_delegates_attributes(self):
        engine = RecordingEngine()
        image = object()
        proxy = GroundedQwenEngine(engine, image)
        with patch(
            "lib_krea2_edit.conditioning.grounded_token_multiplier_compat",
            return_value=nullcontext(),
        ):
            proxy(["prompt"])
        self.assertEqual(engine.calls, [(["prompt"], [image])])
        self.assertEqual(proxy.marker, "delegated")

    def test_setup_swap_is_scoped_and_idempotent(self):
        engine = RecordingEngine()
        image = object()
        processing = FakeProcessing(engine)
        install_grounded_setup(processing, image)
        install_grounded_setup(processing, object())
        with (
            patch(
                "lib_krea2_edit.conditioning.forge_qwen_vision_attention_compat",
                return_value=nullcontext(),
            ),
            patch(
                "lib_krea2_edit.conditioning.grounded_token_multiplier_compat",
                return_value=nullcontext(),
            ),
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
