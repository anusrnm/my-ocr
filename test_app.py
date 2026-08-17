import unittest
from pathlib import Path

import easyocr
import pandas as pd
from pandas.testing import assert_frame_equal

from app import extract_tables


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