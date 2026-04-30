import pytesseract
from PIL import Image
import os
import re

os.environ['PATH'] = r'C:\Program Files\Tesseract-OCR' + os.pathsep + os.environ.get('PATH', '')

# Extract text
img = Image.open('d:/formautofill/pdf/VRAJSHOP.jpeg')
text = pytesseract.image_to_string(img)

# Detect format
if re.search(r'JJJJ|DDDDEEEELLLLIIIIVVVVEEEERRRRYYYY', text):
    fmt = 'A'
elif re.search(r'MILL\s+CHALLAN', text, re.IGNORECASE):
    fmt = 'B'
elif re.search(r'Srl\.?\s*No\.?\s+Taka\s+No', text, re.IGNORECASE):
    fmt = 'C'
else:
    fmt = 'A'

print(f'Detected format: {fmt}')
print()

# Check for specific patterns
print('=== Pattern Matching Results ===')
print(f'Has DELIVERY CHALLAN: {bool(re.search(r"DELIVERY\s+CHALLAN", text, re.IGNORECASE))}')
print(f'Has MILL CHALLAN: {bool(re.search(r"MILL\s+CHALLAN", text, re.IGNORECASE))}')
print(f'Has Srl No Taka No: {bool(re.search(r"Srl\.?\s*No\.?\s+Taka\s+No", text, re.IGNORECASE))}')
print(f'Has Sr Taka No: {bool(re.search(r"Sr.*Taka.*No", text, re.IGNORECASE))}')
print()

# Test challan number extraction
print('=== Challan Number Patterns ===')
patterns = [
    (r'Challan\s+No[.:\s]+(\d+)', 'Pattern 1'),
    (r'Challan\s+No[:\s]*\(?(\d)', 'Pattern 2'),
    (r'Challan\s+[Nn]o[^0-9]*(\d+)', 'Pattern 3'),
    (r'Challan.*?[Nn]o.*?(\d+)', 'Pattern 4 (loose)'),
]

for pat, name in patterns:
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        print(f'{name}: {m.group(1)}')

print()
print('=== Total Meter Patterns ===')
meter_patterns = [
    (r'Total\s+Meters?\s*[:\s]+([0-9,.]+)', 'Pattern 1'),
    (r'Meters?\s*[:\s]+([0-9,.]+)(?:\s|$)', 'Pattern 2'),
    (r'[Mm]eters?\s*[:\s.]+([0-9,.]+)', 'Pattern 3'),
]

for pat, name in meter_patterns:
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        print(f'{name}: {m.group(1)}')

print()
print('=== Quality Patterns ===')
q_patterns = [
    (r'Qlty\s*[:\s]+([^\n]+?)\s*(?:\n|$)', 'Pattern 1 (Qlty)'),
    (r'Quality\s*[:\s]+([^\n]+)', 'Pattern 2 (Quality)'),
    (r'[Qq]lty[:\s]+([^\n\|]+)', 'Pattern 3 (loose Qlty)'),
]

for pat, name in q_patterns:
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        print(f'{name}: {m.group(1)}')

# Show the actual text around "Challan No" and "Total"
print()
print('=== Relevant Text Sections ===')
for line in text.split('\n'):
    if 'Challan' in line or 'Total' in line or 'Qlty' in line:
        print(repr(line))
