import os, pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from extractor import clean_ocr_text

os.environ['PATH'] = r'C:\Program Files\Tesseract-OCR' + os.pathsep + os.environ.get('PATH', '')

def _preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert('L')
    if image.width < 1800:
        scale = 1800 / image.width
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    return image

image = Image.open('pdf/VRAJSHOP.jpeg')
prep_image = _preprocess_image(image)
text = pytesseract.image_to_string(prep_image)
with open('prep_out.txt', 'w', encoding='utf-8') as f:
    f.write(clean_ocr_text(text))
