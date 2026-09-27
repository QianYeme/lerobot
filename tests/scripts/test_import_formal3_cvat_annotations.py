import unittest
import xml.etree.ElementTree as ET

from scripts.import_formal3_cvat_annotations import canonical_xml, validate_record


def box(frame: int, *, keyframe: str = "0", outside: str = "0", xbr: str = "20") -> ET.Element:
    return ET.Element("box", {
        "frame": str(frame), "keyframe": keyframe, "outside": outside, "occluded": "0",
        "xtl": "1", "ytl": "2", "xbr": xbr, "ybr": "30",
    })


class ImportFormal3CvatTests(unittest.TestCase):
    def test_validation_and_canonical_xml(self):
        record = {"size": 2, "tracks": [ET.Element("track", {"label": "cup"})],
                  "boxes": [box(0, keyframe="1"), box(1)]}
        result = validate_record(record, episode=0, length=2, width=640, height=480)
        self.assertEqual(result["active_frames"], 2)
        self.assertEqual(result["keyframes"], 1)
        self.assertEqual(result["interpolated_active_frames"], 1)
        root = ET.fromstring(canonical_xml(result["active_by_frame"]))
        self.assertIsNone(root.find("meta"))
        self.assertEqual([node.get("frame") for node in root.findall(".//box")], ["0", "1"])

    def test_duplicate_and_invalid_boxes_are_rejected(self):
        base = {"size": 2, "tracks": [ET.Element("track", {"label": "cup"})]}
        with self.assertRaises(ValueError):
            validate_record(base | {"boxes": [box(0), box(0)]}, episode=0, length=2, width=640, height=480)
        with self.assertRaises(ValueError):
            validate_record(base | {"boxes": [box(0, xbr="700")]}, episode=0, length=2, width=640, height=480)


if __name__ == "__main__":
    unittest.main()
