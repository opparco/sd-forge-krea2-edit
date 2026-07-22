import unittest

import torch

from lib_krea2_edit.image_utils import fit_reference_pixels, limit_long_side


class GeometryTests(unittest.TestCase):
    def test_near_aspect_ratio_fills_target(self):
        image = torch.zeros(1, 768, 512, 3)
        output = fit_reference_pixels(image, 1024, 704)
        self.assertEqual(tuple(output.shape), (1, 1024, 704, 3))

    def test_wide_aspect_mismatch_is_fit_and_snapped(self):
        image = torch.zeros(1, 1024, 512, 3)
        output = fit_reference_pixels(image, 512, 1024)
        self.assertEqual(output.shape[1], 512)
        self.assertLess(output.shape[2], 1024)
        self.assertEqual(output.shape[2] % 16, 0)
        self.assertEqual(output.shape[1] % 16, 0)

    def test_grounding_only_downscales(self):
        small = torch.zeros(1, 320, 240, 3)
        self.assertIs(limit_long_side(small, 768), small)
        large = torch.zeros(1, 1200, 600, 3)
        self.assertEqual(tuple(limit_long_side(large, 768).shape), (1, 768, 384, 3))


if __name__ == "__main__":
    unittest.main()
