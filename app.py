from img2table.document import Image
from img2table.ocr import EasyOCR

ocr = EasyOCR(lang=["en"])

doc = Image("./sample/sample1.jpg")
extracted_tables = doc.extract_tables(ocr=ocr, implicit_rows=True, implicit_columns=True, borderless_tables=False)

for table in extracted_tables:
    print(table.df)

