import pytesseract
from PIL import Image
import os
import re

os.environ['PATH'] = r'C:\Program Files\Tesseract-OCR' + os.pathsep + os.environ.get('PATH', '')

img = Image.open('d:/formautofill/pdf/VRAJSHOP.jpeg')
text = pytesseract.image_to_string(img)

print('=== RAW OCR TEXT ===')
print(text)
print('\n' + '='*50)
print('=== PATTERN MATCHING ANALYSIS ===')
print('='*50 + '\n')

# Look for challan number
ch_m = re.search(r'[Cc]hallan\s+[Nn]o[.:\s]+([\w/\-]+)', text, re.IGNORECASE)
print('Challan No pattern match:', ch_m.group(1) if ch_m else 'NOT FOUND')

# Look for total taka
tt_m = re.search(r'[Tt]otal\s+[Tt]aka\s*[:\s]+(\d+)', text, re.IGNORECASE)
print('Total Taka pattern match:', tt_m.group(1) if tt_m else 'NOT FOUND')

# Look for total meters
tm_m = re.search(r'[Tt]otal\s+[Mm]et[a-z]*\s*[:\s.]+([0-9,.]+)', text, re.IGNORECASE)
print('Total Meter pattern match:', tm_m.group(1) if tm_m else 'NOT FOUND')

# Look for date
dt_m = re.search(r'[Cc]hallan\s+[Dd]ate\s*[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
print('Challan Date (slash/dash pattern):', dt_m.group(1) if dt_m else 'NOT FOUND')

# Try finding just the year
dt_m2 = re.search(r'[Dd]ate\s*[:\s]+(\d{4})', text, re.IGNORECASE)
print('Date (year pattern):', dt_m2.group(1) if dt_m2 else 'NOT FOUND')

# Quality
q_m = re.search(r'[Qq]uality[:\s*]+([A-Z0-9\s\-\.\/]+?)(?:\s+[A-Z]{3}|$)', text, re.IGNORECASE)
print('Quality pattern match:', q_m.group(1) if q_m else 'NOT FOUND')

# HSN Code
hsn_patterns = [
    r'HSN\s*/\s*SAC\s*[:\s]+(\d+)',
    r'HSN\s+[A-Z]+\s*[:\s]+(\d+)',
    r'\bHSN\s*[:\s]+(\d+)',
]
for pat in hsn_patterns:
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        print(f'HSN Code found with pattern {pat}: {m.group(1)}')
        break
else:
    print('HSN Code: NOT FOUND')
