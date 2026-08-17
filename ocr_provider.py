"""Provider-neutral OCR results and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class OcrWord:
    """One recognized word and its quadrilateral coordinates."""

    box: tuple[tuple[float, float], ...]
    text: str
    confidence: float


class OcrProvider(Protocol):
    """Interface shared by OCR engines used by the table parser."""

    def read(self, image: np.ndarray) -> Sequence[OcrWord]:
        """Recognize text in a BGR image."""


class EasyOcrProvider:
    """Adapt an initialized EasyOCR reader to the provider contract."""

    def __init__(self, reader: Any) -> None:
        self.reader = reader

    def read(self, image: np.ndarray) -> list[OcrWord]:
        results = self.reader.readtext(
            image,
            detail=1,
            mag_ratio=1.5,
            contrast_ths=0.1,
            adjust_contrast=0.6,
            decoder="beamsearch",
            beamWidth=5,
        )
        return [
            OcrWord(
                tuple((float(point[0]), float(point[1])) for point in box),
                str(text),
                float(confidence),
            )
            for box, text, confidence in results
        ]


class TesseractOcrProvider:
    """Adapt pytesseract word data to the provider contract."""

    def __init__(self) -> None:
        import pytesseract

        self.pytesseract = pytesseract

    def read(self, image: np.ndarray) -> list[OcrWord]:
        data = self.pytesseract.image_to_data(
            image,
            output_type=self.pytesseract.Output.DICT,
            lang="eng",
        )
        words: list[OcrWord] = []
        for index, text in enumerate(data["text"]):
            text = str(text).strip()
            confidence = float(data["conf"][index])
            if not text or confidence < 0:
                continue

            x = float(data["left"][index])
            y = float(data["top"][index])
            width = float(data["width"][index])
            height = float(data["height"][index])
            words.append(
                OcrWord(
                    ((x, y), (x + width, y), (x + width, y + height), (x, y + height)),
                    text,
                    confidence / 100.0,
                )
            )
        return words