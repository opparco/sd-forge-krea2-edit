import unittest

import torch

from lib_krea2_edit.image_utils import (
    fit_reference_mask,
    fit_reference_pixels,
    limit_long_side,
    order_reference_images,
)


class GeometryTests(unittest.TestCase):
    def test_subject_only_reference_order(self):
        subject = object()
        self.assertEqual(order_reference_images(subject), [subject])

    def test_scene_subject_reference_order(self):
        subject = object()
        scene = object()
        self.assertEqual(order_reference_images(subject, scene), [scene, subject])

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

    def test_boost_mask_uses_reference_fit_geometry(self):
        image = torch.zeros(1, 1024, 512, 3)
        mask = torch.ones(1, 1024, 512)
        fitted_image = fit_reference_pixels(image, 512, 1024)
        fitted_mask = fit_reference_mask(mask, image, 512, 1024)
        self.assertEqual(tuple(fitted_mask.shape[1:]), tuple(fitted_image.shape[1:3]))


if __name__ == "__main__":
    unittest.main()
