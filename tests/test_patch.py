import copy
import unittest

import torch
import torch.nn.functional as F

from backend.nn.krea import SingleStreamDiT
from lib_krea2_edit.patch import patch_krea2_unet


class IdentityPredictor:
    def calculate_input(self, sigma, value):
        return value

    def timestep(self, sigma):
        return sigma

    def calculate_denoised(self, sigma, model_output, original):
        return model_output


class FakeKModel:
    def __init__(self):
        self.diffusion_model = SingleStreamDiT(
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
        self.predictor = IdentityPredictor()
        self.computation_dtype = torch.float32


class FakeUnet:
    def __init__(self, previous_wrapper=None):
        self.model = FakeKModel()
        self.model_options = {"transformer_options": {}}
        if previous_wrapper is not None:
            self.model_options["model_function_wrapper"] = previous_wrapper

    def clone(self):
        result = object.__new__(FakeUnet)
        result.model = self.model
        result.model_options = copy.copy(self.model_options)
        result.model_options["transformer_options"] = copy.copy(
            self.model_options["transformer_options"]
        )
        return result

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


class FakeLatentFormat:
    @staticmethod
    def process_in(latent):
        return latent


class FakeVAE:
    first_stage_model = FakeLatentFormat()

    @staticmethod
    def encode(image):
        channels_first = image.movedim(-1, 1)
        channels_first = torch.cat(
            (channels_first, torch.zeros_like(channels_first[:, :1])), dim=1
        )
        return F.interpolate(channels_first, scale_factor=1 / 8, mode="area")


class PatchTests(unittest.TestCase):
    def test_existing_model_wrapper_is_chained(self):
        calls = []

        def previous_wrapper(model_function, call):
            calls.append("previous")
            return model_function(call["input"], call["timestep"], **call["c"])

        unet = FakeUnet(previous_wrapper)
        patched = patch_krea2_unet(unet, FakeVAE(), torch.zeros(1, 64, 64, 3))
        wrapper = patched.model_options["model_function_wrapper"]
        call = {
            "input": torch.randn(1, 4, 8, 8),
            "timestep": torch.ones(1),
            "c": {
                "c_crossattn": torch.randn(1, 1, 3, 32),
                "transformer_options": {},
            },
        }

        with torch.inference_mode():
            output = wrapper(None, call)

        self.assertEqual(calls, ["previous"])
        self.assertEqual(tuple(output.shape), tuple(call["input"].shape))


if __name__ == "__main__":
    unittest.main()
