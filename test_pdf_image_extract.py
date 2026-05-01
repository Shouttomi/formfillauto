from pathlib import Path
import shutil

from PIL import Image

import extractor


def test_scanned_pdf_falls_back_to_ocr():
    source_image = Path("pdf/VRAJSHOP.jpeg")
    assert source_image.exists(), f"Missing fixture: {source_image}"

    temp_dir = Path("pdf") / "_tmp_test_pdf_image_extract"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    original_ocr = extractor._extract_challan_from_ocr_image
    try:
        extractor._extract_challan_from_ocr_image = lambda image: {
            "challan_no": "OCR-TEST-1",
            "party": "OCR TEST PARTY",
            "date": "01/05/2026",
            "quality": "TEST QUALITY",
            "meter": 123.0,
            "table": [{"srl_no": 1, "meter": 123.0}],
        }

        pdf_path = temp_dir / "scanned.pdf"
        Image.open(source_image).convert("RGB").save(pdf_path, "PDF", resolution=200.0)

        challans = extractor.extract_challans(str(pdf_path))
    finally:
        extractor._extract_challan_from_ocr_image = original_ocr
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    assert challans, "Expected OCR fallback to return at least one challan for scanned PDF"
    challan = challans[0]
    assert challan.get("challan_no") == "OCR-TEST-1", challan
    assert challan.get("table"), challan


if __name__ == "__main__":
    test_scanned_pdf_falls_back_to_ocr()
    print("test_scanned_pdf_falls_back_to_ocr passed")
