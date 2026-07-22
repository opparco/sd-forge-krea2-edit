import unittest
from unittest.mock import patch

import torch

from backend.attention import attention_pytorch
from backend.nn.krea import SingleStreamDiT
from lib_krea2_edit.forward import krea2_edit_forward, reference_attention_bias


class ForwardShapeTests(unittest.TestCase):
    @staticmethod
    def make_model():
        model = SingleStreamDiT(
            features=32,
            tdim=8,
            txtdim=16,
            heads=2,
            kvheads=1,
            multiplier=1,
            layers=1,
            patch=2,
            channels=4,
            txtlayers=2,
            txtheads=2,
            txtkvheads=1,
        ).eval()
        # Krea modulation parameters use torch.empty and need deterministic test
        # initialization outside the real checkpoint loader.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0.0, 0.02)
        return model

    def test_single_reference_returns_only_target_shape(self):
        model = self.make_model()
        target = torch.randn(1, 4, 8, 12)
        source = torch.randn(1, 4, 8, 8)
        context = torch.randn(1, 1, 3, 32)
        timestep = torch.ones(1)

        with torch.inference_mode():
            output = krea2_edit_forward(model, target, timestep, context, source, {})

        self.assertEqual(tuple(output.shape), tuple(target.shape))

    def test_two_references_return_only_target_shape(self):
        model = self.make_model()
        target = torch.randn(1, 4, 8, 12)
        sources = [torch.randn(1, 4, 8, 8), torch.randn(1, 4, 6, 10)]
        context = torch.randn(1, 1, 3, 32)

        with torch.inference_mode():
            output = krea2_edit_forward(
                model, target, torch.ones(1), context, sources, {}
            )

        self.assertEqual(tuple(output.shape), tuple(target.shape))

    def test_boost_one_is_numerical_no_op(self):
        model = self.make_model()
        target = torch.randn(1, 4, 8, 8)
        source = torch.randn(1, 4, 8, 8)
        context = torch.randn(1, 1, 3, 32)
        with (
            patch("backend.nn.krea.attention_function", attention_pytorch),
            torch.inference_mode(),
        ):
            baseline = krea2_edit_forward(
                model, target, torch.ones(1), context, source, {}
            )
            boosted_defaults = krea2_edit_forward(
                model,
                target,
                torch.ones(1),
                context,
                source,
                {},
                ref_boost=1.0,
                ref_boost_a=1.0,
            )
        self.assertTrue(
            torch.allclose(baseline, boosted_defaults, atol=1e-6, rtol=1e-6)
        )

    def test_reference_attention_bias_targets_expected_reference_columns(self):
        bias = reference_attention_bias(
            [2.0, 3.0],
            None,
            text_length=2,
            source_lengths=[4, 2],
            target_length=3,
            source_grids=[(2, 2), (1, 2)],
            device="cpu",
            dtype=torch.float32,
        )
        self.assertTrue(torch.all(bias[:, :, :8] == 0))
        self.assertTrue(
            torch.allclose(bias[:, :, 8:, 2:6], torch.log(torch.tensor(2.0)))
        )
        self.assertTrue(
            torch.allclose(bias[:, :, 8:, 6:8], torch.log(torch.tensor(3.0)))
        )

    def test_black_mask_disables_last_reference_boost(self):
        bias = reference_attention_bias(
            [4.0],
            torch.zeros(1, 8, 8),
            text_length=2,
            source_lengths=[4],
            target_length=3,
            source_grids=[(2, 2)],
            device="cpu",
            dtype=torch.float32,
        )
        self.assertTrue(torch.count_nonzero(bias) == 0)

    def test_boost_runs_through_pytorch_attention(self):
        model = self.make_model()
        target = torch.randn(1, 4, 8, 8)
        sources = [torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)]
        context = torch.randn(1, 1, 3, 32)
        with (
            patch("backend.nn.krea.attention_function", attention_pytorch),
            torch.inference_mode(),
        ):
            output = krea2_edit_forward(
                model,
                target,
                torch.ones(1),
                context,
                sources,
                {},
                ref_boost=2.0,
                ref_boost_a=1.5,
                ref_boost_mask=torch.ones(1, 8, 8),
            )
        self.assertEqual(tuple(output.shape), tuple(target.shape))


if __name__ == "__main__":
    unittest.main()
