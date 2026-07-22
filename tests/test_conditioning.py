import unittest

from lib_krea2_edit.conditioning import GroundedQwenEngine, install_grounded_setup


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
        processing.setup_conds()
        self.assertIs(processing.sd_model.text_processing_engine_qwen, engine)
        self.assertEqual(engine.calls, [(["positive", "negative"], [image])])


if __name__ == "__main__":
    unittest.main()
