"""Extract invoice tables from photos using EasyOCR word boxes.

img2table's layout detection fails on these scans because the printed value block
drifts vertically relative to the text block, so rows are rebuilt from OCR
geometry instead: header words define the columns, and each column is aligned
independently.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import pymupdf

MIN_CONFIDENCE = 0.15
SECTION_KEYWORDS = ("invoice",)


@dataclass
class Word:
    x1: int
    y1: int
    x2: int
    y2: int
    text: str

    @property
    def xc(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def yc(self) -> float:
        return (self.y1 + self.y2) / 2


def crop_page(img: np.ndarray) -> np.ndarray:
    """Perspective-crop the sheet of paper out of the photo background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    page = max(contours, key=cv2.contourArea)
    if cv2.contourArea(page) < 0.3 * img.shape[0] * img.shape[1]:
        return img

    approx = cv2.approxPolyDP(page, 0.02 * cv2.arcLength(page, True), True)
    pts = (
        approx.reshape(4, 2).astype(np.float32)
        if len(approx) == 4
        else cv2.boxPoints(cv2.minAreaRect(page)).astype(np.float32)
    )
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    tl, tr, br, bl = pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    src = np.array([tl, tr, br, bl], np.float32)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))


def deskew(img: np.ndarray, limit: float = 6.0, step: float = 0.1) -> np.ndarray:
    """Rotate so text rows are horizontal, by maximising horizontal projection contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 900 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15
    )
    center = (binary.shape[1] / 2, binary.shape[0] / 2)

    def sharpness(angle: float) -> float:
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(binary, m, (binary.shape[1], binary.shape[0]), flags=cv2.INTER_NEAREST)
        return float(np.sum(np.diff(rot.sum(axis=1, dtype=np.float64)) ** 2))

    angle = float(max(np.arange(-limit, limit + step, step), key=sharpness))
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def read_words(reader: easyocr.Reader, img: np.ndarray) -> list[Word]:
    words = []
    for box, text, conf in reader.readtext(img, detail=1):
        if conf < MIN_CONFIDENCE or not text.strip():
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append(Word(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), text.strip()))
    return sorted(words, key=lambda w: (w.y1, w.x1))


def detect_header(words: list[Word]) -> list[Word]:
    """The topmost group of words sharing a baseline defines the column layout."""
    if not words:
        return []
    tolerance = np.median([w.y2 - w.y1 for w in words])
    first = words[0]
    header = [w for w in words if abs(w.yc - first.yc) <= tolerance]
    return sorted(header, key=lambda w: w.x1)


def column_bounds(header: list[Word]) -> list[float]:
    edges = [(prv.x2 + nxt.x1) / 2 for prv, nxt in zip(header, header[1:])]
    return [-np.inf, *edges, np.inf]


def split_sections(words: list[Word], header: list[Word]) -> list[tuple[str, list[Word]]]:
    """Split rows on band titles such as 'Part Invoice ...' / 'Labour Invoice ...'."""
    if not header:
        return []
    body = [w for w in words if w.yc > max(h.y2 for h in header)]
    if not body:
        return []
    line_height = float(np.median([w.y2 - w.y1 for w in body]))

    markers = [w for w in body if any(k in w.text.lower() for k in SECTION_KEYWORDS)]
    sections: list[tuple[str, list[Word]]] = []
    starts = [(w.yc, w) for w in markers] or [(-np.inf, None)]
    for i, (y, marker) in enumerate(starts):
        y_end = starts[i + 1][0] if i + 1 < len(starts) else np.inf
        band = [w for w in body if y - line_height <= w.yc <= y + line_height] if marker else []
        title = " ".join(w.text for w in sorted(band, key=lambda w: w.x1)) if marker else "table"
        rows = [
            w
            for w in body
            if y + line_height < w.yc < y_end - line_height and w not in band
        ]
        if rows:
            sections.append((title, rows))
    return sections


def align_column(items: list[Word], anchors: list[float], ordinal: bool) -> list[str]:
    """Fit a column's items into one cell per row."""
    items = sorted(items, key=lambda w: w.yc)
    groups: list[list[Word]] = [[] for _ in anchors]

    if ordinal:
        for i, w in enumerate(items):
            groups[i].append(w)
    else:
        # Ragged column: snap to the nearest row, so wrapped lines join their own row.
        for w in items:
            groups[int(np.argmin([abs(a - w.yc) for a in anchors]))].append(w)

    return [" ".join(w.text for w in sorted(g, key=lambda w: (w.y1, w.x1))) for g in groups]


def assign_columns(words: list[Word], header: list[Word], bounds: list[float]) -> list[list[Word]]:
    """Split words into columns, then refine using the horizontal span of each column."""
    columns: list[list[Word]] = [[] for _ in header]
    for w in words:
        idx = int(np.searchsorted(bounds, w.xc, side="right")) - 1
        columns[min(max(idx, 0), len(header) - 1)].append(w)

    spans = [
        (min([h.x1, *[w.x1 for w in col]]), max([h.x2, *[w.x2 for w in col]]))
        for h, col in zip(header, columns)
    ]
    refined: list[list[Word]] = [[] for _ in header]
    for w in words:
        overlaps = [min(w.x2, x2) - max(w.x1, x1) for x1, x2 in spans]
        best = int(np.argmax(overlaps))
        if overlaps[best] <= 0:
            best = min(max(int(np.searchsorted(bounds, w.xc, side="right")) - 1, 0), len(header) - 1)
        refined[best].append(w)
    return refined


def row_model(
    columns: list[list[Word]], n_rows: int
) -> tuple[np.ndarray, np.ndarray, set[int]]:
    """Fit each row's vertical position as a function of column position.

    The two printed blocks of these invoices use slightly different line pitches, so
    every column gets its own (pitch, offset), interpolated from the columns that hold
    exactly one item per row.
    """
    rows = np.arange(n_rows)
    fits: dict[int, tuple[float, float, float]] = {}
    for i, col in enumerate(columns):
        if len(col) != n_rows or n_rows < 2:
            continue
        ys = np.array([w.yc for w in sorted(col, key=lambda w: w.yc)])
        (pitch, offset), (residual, *_) = np.polyfit(rows, ys, 1, full=True)[:2]
        fits[i] = (pitch, offset, float(np.sqrt(residual / n_rows)))

    trusted = {i: f for i, f in fits.items() if f[2] < 0.3 * f[0]} or fits
    if not trusted:
        longest = sorted(max(columns, key=len), key=lambda w: w.yc)
        ys = [w.yc for w in longest][:n_rows] or [0.0]
        pitch = (ys[-1] - ys[0]) / max(len(ys) - 1, 1)
        return np.array([0.0, pitch]), np.array([0.0, ys[0]]), set()

    xs = [float(np.median([w.xc for w in columns[i]])) for i in trusted]
    deg = 1 if len(xs) > 1 else 0
    pitch_by_x = np.polyfit(xs, [f[0] for f in trusted.values()], deg)
    offset_by_x = np.polyfit(xs, [f[1] for f in trusted.values()], deg)
    return pitch_by_x, offset_by_x, set(trusted)


def build_dataframe(rows: list[Word], header: list[Word], bounds: list[float]) -> pd.DataFrame:
    columns = assign_columns(rows, header, bounds)
    counts = [len(c) for c in columns if c]
    if not counts:
        return pd.DataFrame()

    n_rows = Counter(counts).most_common(1)[0][0]
    pitch_by_x, offset_by_x, trusted = row_model(columns, n_rows)

    cell_columns = []
    for i, (h, col) in enumerate(zip(header, columns)):
        x = float(np.median([w.xc for w in col])) if col else h.xc
        pitch, offset = np.polyval(pitch_by_x, x), np.polyval(offset_by_x, x)
        anchors = [float(offset + pitch * r) for r in range(n_rows)]
        cell_columns.append(align_column(col, anchors, ordinal=i in trusted))
    return pd.DataFrame(np.array(cell_columns, dtype=object).T, columns=[h.text for h in header])


def render_pdf_pages(path: str, dpi: int = 200) -> list[np.ndarray]:
    """Render PDF pages to BGR images for the existing OCR pipeline."""
    try:
        document = pymupdf.open(path)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Unable to read PDF: {path}") from exc

    if document.page_count == 0:
        document.close()
        raise ValueError(f"PDF has no pages: {path}")

    scale = dpi / 72
    matrix = pymupdf.Matrix(scale, scale)
    pages = []
    try:
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8)
            image = image.reshape(pixmap.height, pixmap.width, pixmap.n)
            pages.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    finally:
        document.close()
    return pages


def extract_tables_from_image(
    source: np.ndarray, reader: easyocr.Reader
) -> list[tuple[str, pd.DataFrame]]:
    img = deskew(crop_page(source))
    words = read_words(reader, img)
    if not words:
        return []
    header = detect_header(words)
    bounds = column_bounds(header)
    return [
        (title, build_dataframe(rows, header, bounds))
        for title, rows in split_sections(words, header)
    ]


def extract_tables(path: str, reader: easyocr.Reader) -> list[tuple[str, pd.DataFrame]]:
    if Path(path).suffix.lower() == ".pdf":
        tables = []
        for page_number, image in enumerate(render_pdf_pages(path), start=1):
            for title, dataframe in extract_tables_from_image(image, reader):
                tables.append((f"page {page_number}: {title}", dataframe))
        return tables

    source = cv2.imread(path)
    if source is None:
        raise ValueError(f"Unable to read image: {path}")
    return extract_tables_from_image(source, reader)


def expand_paths(patterns: list[str]) -> list[str]:
    paths = []
    for pattern in patterns:
        matches = sorted(glob(pattern, recursive=True))
        paths.extend(matches or [pattern])
    return paths


def main(paths: list[str]) -> None:
    reader = easyocr.Reader(["en"])
    for path in paths:
        for title, df in extract_tables(path, reader):
            print(f"\n=== {path} :: {title} ===")
            print(df.to_string(index=False))


if __name__ == "__main__":
    main(expand_paths(sys.argv[1:] or ["./sample/sample1.jpg"]))
