import json
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from check_c50_sam_contour import contour_mask


class ContourTests(unittest.TestCase):
    def test_outline_fill_and_json_metadata(self):
        image = Image.new("RGB", (640,480), "gray")
        ImageDraw.Draw(image).rectangle((100,200,140,270), outline="red", width=2)
        mask, details = contour_mask(image)
        self.assertTrue(mask[235,120])
        self.assertFalse(mask[190,120])
        self.assertFalse(mask[235,150])
        json.dumps(details)

    def test_missing_outline_rejected(self):
        with self.assertRaises(ValueError):
            contour_mask(Image.new("RGB", (640,480), "gray"))


if __name__ == "__main__":
    unittest.main()
