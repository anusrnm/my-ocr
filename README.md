# Invoice OCR

Extract invoice tables from photos and scanned PDF files using EasyOCR or Tesseract and OpenCV.
The table rows are reconstructed from OCR word positions so the extractor can handle
invoice layouts where the description and numeric columns drift vertically.

## Features

- Reads JPG, PNG, and other formats supported by OpenCV.
- Reads scanned PDFs by rendering each page with PyMuPDF.
- Crops and deskews photographed pages.
- Detects table headers and reconstructs columns from word geometry.
- Splits invoices into sections such as parts and labour.
- Processes multiple files and glob patterns in one command.

## Requirements

- Python 3.10 or newer
- CPU or CUDA-capable PyTorch installation
- Dependencies from `requirements.txt`

Install the dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Docker

The included CPU image runs the same CLI with a persistent EasyOCR model cache.
On Windows, use Docker from the Ubuntu WSL2 distribution. From PowerShell, build
the image through WSL:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/mywork/my-ocr && docker build -t invoice-ocr ."
```

Process a host image with EasyOCR by mounting the input directory and a model-cache
directory through WSL:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "docker run --rm \
  -v /mnt/c/mywork/my-ocr/sample:/work:ro \
  -v /mnt/c/mywork/my-ocr/.models:/models \
  invoice-ocr /work/sample1.jpg"
```

  Run the same image with Tesseract:

  ```powershell
  wsl -d Ubuntu-24.04 -- bash -lc "docker run --rm \
    -v /mnt/c/mywork/my-ocr/sample:/work:ro \
    invoice-ocr --provider tesseract /work/sample1.jpg"
  ```

  The image contains both providers. EasyOCR is the default; select Tesseract with
  `--provider tesseract` and capture each output separately to compare recognition:

  ```powershell
  wsl -d Ubuntu-24.04 -- bash -lc "docker run --rm \
    -v /mnt/c/mywork/my-ocr/sample:/work:ro \
    -v /mnt/c/mywork/my-ocr/.models:/models \
    invoice-ocr /work/sample1.jpg" | Out-File -Encoding utf8 easyocr-output.txt

  wsl -d Ubuntu-24.04 -- bash -lc "docker run --rm \
    -v /mnt/c/mywork/my-ocr/sample:/work:ro \
    invoice-ocr --provider tesseract /work/sample1.jpg" | Out-File -Encoding utf8 tesseract-output.txt
  ```

The first run downloads the English model into `.models`; later runs reuse it.
Ubuntu WSL2 with Docker Engine is the supported Windows deployment path.
EasyOCR may download its English recognition model the first time it runs. OCR uses
CPU automatically when CUDA or MPS is unavailable.

## Usage

Process one image:

```powershell
python app.py sample/sample1.jpg
```

Select Tesseract instead of the default EasyOCR provider:

```powershell
python app.py --provider tesseract sample/sample1.jpg
```

Process a PDF:

```powershell
python app.py invoice.pdf
```

Process several files:

```powershell
python app.py sample/sample1.jpg sample/sample2.jpg invoice.pdf
```

Use a glob pattern in PowerShell by quoting it:

```powershell
python app.py "sample/*.jpg"
```

With no arguments, the program processes `./sample/sample1.jpg`.

## Output

Results are printed to the terminal. Each table is labeled with its source path and,
for PDF input, its page number. Enhanced PNG copies are not saved by default. Enable
offline review images under the ignored `./.enhanced/` folder with:

```powershell
python app.py --save-enhanced invoice.pdf
```

Use `--no-save-enhanced` to explicitly disable saving:

```text
=== invoice.pdf :: page 1: Part Invoice ... ===
```

The current program does not write CSV or JSON files. Redirect terminal output when
a text capture is useful:

```powershell
python app.py invoice.pdf | Out-File -Encoding utf8 invoice-output.txt
```

## Processing pipeline

1. Load an image directly or render each PDF page at 200 DPI.
2. Detect and perspective-crop the paper where possible.
3. Deskew the page.
4. Run the selected OCR provider and discard words below the configured confidence threshold.
5. Use the first OCR word band as the table header.
6. Assign body words to columns and align rows using their vertical positions.

## Limitations

- This is tuned for invoice-like tabular documents and English OCR.
- OCR output can contain recognition errors for blurred text, symbols, commas, and
  low-resolution scans.
- PDFs are processed as images, including text-based PDFs; embedded PDF text is not
  extracted directly.
- Output is currently intended for review in the terminal rather than direct accounting
  import.
