"""Verify independent prompts reset tracking; retain default propagation behavior."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import prepare_c50_mask_preview as preview


class Predictor:
    def __init__(self):
        self.resets = 0
        self.prompts = 0
        self.propagations = 0
        self.logits = torch.zeros(1, 1, 480, 640)

    def init_state(self, **kwargs):
        return {}

    def reset_state(self, state):
        state.clear()
        self.resets += 1

    def add_new_points_or_box(self, inference_state, frame_idx, **kwargs):
        if self.resets:
            assert not inference_state
        inference_state["tracking"] = True
        self.prompts += 1
        return frame_idx, [1], self.logits

    def propagate_in_video(self, state):
        self.propagations += 1
        for index in range(718):
            yield index, [1], self.logits


class PreviewStrategyTests(unittest.TestCase):
    def check_strategy(self, independent):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / "labels.xml"
            xml.write_text("fixture")
            labels = {index: [100, 100, 200, 200] for index in range(718)}
            def extract(video, start, length, fps, frames):
                for index in [0, 120, 240, 360, 480, 717]:
                    Image.new("RGB", (640, 480)).save(frames / f"{index:06d}.jpg")
                return [index / fps for index in range(length)]
            predictor = Predictor()
            with patch.object(preview, "boxes", return_value=(labels, xml)), \
                 patch.object(preview, "video_mapping", return_value=(root / "fixture.mp4", 0)), \
                 patch.object(preview, "extract_frames", side_effect=extract), \
                 patch("builtins.print"):
                preview.generate(root, {"episode_index": 0, "length": 718}, root,
                                 predictor, 30, negative_margin=25, framewise_box=independent)
            self.assertEqual(predictor.resets, 718 if independent else 0)
            self.assertEqual(predictor.prompts, 718 if independent else 1)
            self.assertEqual(predictor.propagations, 0 if independent else 1)
            record = json.loads((root / "episode_000.json").read_text())
            self.assertTrue(record["generation_complete"])
            self.assertEqual(len(record["prompts"]), predictor.prompts)

    def test_independent_frames_reset_tracking(self):
        self.check_strategy(True)

    def test_default_uses_video_propagation(self):
        self.check_strategy(False)


if __name__ == "__main__":
    unittest.main()
