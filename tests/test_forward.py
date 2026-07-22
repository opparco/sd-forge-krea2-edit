import unittest

import torch

from backend.nn.krea import SingleStreamDiT
from lib_krea2_edit.forward import krea2_edit_forward


class ForwardShapeTests(unittest.TestCase):
    def test_single_reference_returns_only_target_shape(self):
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
        target = torch.randn(1, 4, 8, 12)
        source = torch.randn(1, 4, 8, 8)
        context = torch.randn(1, 1, 3, 32)
        timestep = torch.ones(1)

        with torch.inference_mode():
            output = krea2_edit_forward(model, target, timestep, context, source, {})

        self.assertEqual(tuple(output.shape), tuple(target.shape))


if __name__ == "__main__":
    unittest.main()
