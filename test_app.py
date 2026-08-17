import unittest
from pathlib import Path

import easyocr
import pandas as pd
from pandas.testing import assert_frame_equal

from app import extract_tables, normalize_ocr_text, parse_args


class OcrTextNormalizationTest(unittest.TestCase):
    def test_repairs_invoice_punctuation_artifacts(self):
        self.assertEqual(normalize_ocr_text("] LITER-PRE MIX"), "1 LITRE-PRE MIX")
        self.assertEqual(normalize_ocr_text("18%/"), "18%")

    def test_enhanced_image_saving_is_opt_in(self):
        self.assertFalse(parse_args(["sample/sample1.jpg"]).save_enhanced)
        self.assertTrue(parse_args(["--save-enhanced", "sample/sample1.jpg"]).save_enhanced)
        self.assertFalse(parse_args(["--no-save-enhanced"]).save_enhanced)


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