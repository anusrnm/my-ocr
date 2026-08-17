import unittest
import sys
from unittest.mock import patch
from pathlib import Path

import easyocr
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from app import extract_tables, normalize_ocr_text, parse_args
from ocr_provider import EasyOcrProvider, TesseractOcrProvider


class FakeReader:
    def readtext(self, image, **kwargs):
        return [([[10, 20], [30, 20], [30, 40], [10, 40]], "ABC", 0.91)]


class FakeTesseract:
    class Output:
        DICT = object()

    @staticmethod
    def image_to_data(image, output_type, lang):
        return {
            "text": ["  ABC ", "", "ignored"],
            "conf": ["91.0", "-1", "-1"],
            "left": [10, 0, 1],
            "top": [20, 0, 1],
            "width": [20, 0, 3],
            "height": [20, 0, 4],
        }


class OcrProviderTest(unittest.TestCase):
    def test_easyocr_adapter_returns_provider_result(self):
        result = EasyOcrProvider(FakeReader()).read(np.zeros((40, 40, 3), dtype=np.uint8))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "ABC")
        self.assertEqual(result[0].confidence, 0.91)
        self.assertEqual(result[0].box[0], (10.0, 20.0))

    def test_tesseract_adapter_maps_word_data(self):
        with patch.dict(sys.modules, {"pytesseract": FakeTesseract}):
            result = TesseractOcrProvider().read(np.zeros((40, 40, 3), dtype=np.uint8))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "ABC")
        self.assertEqual(result[0].confidence, 0.91)
        self.assertEqual(result[0].box, ((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)))


class OcrTextNormalizationTest(unittest.TestCase):
    def test_repairs_invoice_punctuation_artifacts(self):
        self.assertEqual(normalize_ocr_text("] LITER-PRE MIX"), "1 LITRE-PRE MIX")
        self.assertEqual(normalize_ocr_text("18%/"), "18%")

    def test_enhanced_image_saving_is_opt_in(self):
        self.assertFalse(parse_args(["sample/sample1.jpg"]).save_enhanced)
        self.assertTrue(parse_args(["--save-enhanced", "sample/sample1.jpg"]).save_enhanced)
        self.assertFalse(parse_args(["--no-save-enhanced"]).save_enhanced)

    def test_provider_defaults_to_easyocr_and_accepts_tesseract(self):
        self.assertEqual(parse_args(["sample/sample1.jpg"]).provider, "easyocr")
        self.assertEqual(parse_args(["--provider", "tesseract"]).provider, "tesseract")


class SampleImageTest(unittest.TestCase):
    def test_sample1_matches_expected_csv(self):
        root = Path(__file__).parent
        reader = easyocr.Reader(["en"])
        tables = extract_tables(str(root / "sample" / "sample1.jpg"), reader)

        actual = pd.concat(
            [dataframe.assign(section=title) for title, dataframe in tables],
            ignore_index=True,
        )
        actual = actual[["section", *[column for column in actual.columns if column != "section"]]]
        expected = pd.read_csv(
            root / "sample" / "sample1_expected.csv",
            dtype=str,
            keep_default_na=False,
        )

        assert_frame_equal(actual, expected, check_dtype=False)


if __name__ == "__main__":
    unittest.main()