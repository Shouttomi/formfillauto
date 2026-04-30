import re
import os
import pdfplumber
import pytesseract
from pytesseract import Output
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Dict, Any


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Enhance image quality before OCR: grayscale, contrast, sharpness, upscale."""
    image = image.convert('L')
    if image.width < 1800:
        scale = 1800 / image.width
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    return image


def _ensure_tesseract_path() -> None:
    tesseract_path = r'C:\Program Files\Tesseract-OCR'
    if tesseract_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = tesseract_path + os.pathsep + os.environ.get('PATH', '')


def _ocr_image(image: Image.Image, config: str = '') -> str:
    _ensure_tesseract_path()
    return pytesseract.image_to_string(image, config=config)


def extract_layout_text_from_image(image_path: str) -> str:
    """Rebuild OCR output into line-oriented text using tesseract word positions."""
    try:
        image = _preprocess_image(Image.open(image_path))
        _ensure_tesseract_path()
        data = pytesseract.image_to_data(image, output_type=Output.DICT, config=r'--oem 3 --psm 6')

        lines = {}
        n = len(data.get('text', []))
        for i in range(n):
            word = (data['text'][i] or '').strip()
            if not word:
                continue
            conf = str(data.get('conf', [''])[i]).strip()
            try:
                if conf and float(conf) < 0:
                    continue
            except ValueError:
                pass

            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            lines.setdefault(key, []).append({
                'left': int(data['left'][i]),
                'width': int(data['width'][i]),
                'text': word,
            })

        rendered_lines = []
        for key in sorted(lines.keys()):
            words = sorted(lines[key], key=lambda item: item['left'])
            if not words:
                continue

            parts = []
            prev_right = None
            for word in words:
                if prev_right is not None:
                    gap = max(1, int((word['left'] - prev_right) / 28))
                    parts.append(' ' * min(gap, 8))
                parts.append(word['text'])
                prev_right = word['left'] + word['width']

            rendered_lines.append(''.join(parts).strip())

        return '\n'.join(rendered_lines)
    except Exception:
        return ""


def extract_table_text_from_image(image_path: str) -> str:
    """Extract text from image specifically optimized for blurry table numbers."""
    try:
        image = _preprocess_image(Image.open(image_path))
        # Use psm 6 (single block) to maximize number extraction in blurry regions
        text = _ocr_image(image, config=r'--oem 3 --psm 6')
        return text
    except Exception:
        return ""


def extract_text_from_image(image_path: str, preprocess: bool = False, config: str = '') -> str:
    """Extract text from JPEG/PNG using Tesseract OCR (default layout)."""
    try:
        image = Image.open(image_path)
        if preprocess:
            image = _preprocess_image(image)
        text = _ocr_image(image, config=config)
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract text from image: {str(e)}")


def _merge_extracted_challans(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing fields in primary using values from secondary."""
    merged = dict(primary)

    for key, value in secondary.items():
        current = merged.get(key)

        if key == 'gstin_map':
            current_map = current if isinstance(current, dict) else {}
            secondary_map = value if isinstance(value, dict) else {}
            if current_map:
                merged[key] = current_map
                continue
            cleaned_secondary = {}
            for map_key, map_value in secondary_map.items():
                normalized_key = re.sub(r'[^A-Z0-9 ]', '', str(map_key).upper()).strip()
                if (
                    map_value
                    and normalized_key
                    and len(normalized_key) >= 6
                    and re.search(r'[A-Z]{3,}', normalized_key)
                ):
                    cleaned_secondary[map_key] = map_value
            merged[key] = {**current_map, **cleaned_secondary}
            continue

        if key == 'table':
            current_table = current if isinstance(current, list) else []
            secondary_table = value if isinstance(value, list) else []
            if len(secondary_table) > len(current_table):
                merged[key] = secondary_table
            continue

        if isinstance(current, str):
            if (not current.strip()) and isinstance(value, str) and value.strip():
                merged[key] = value
        elif isinstance(current, (int, float)):
            if current == 0 and isinstance(value, (int, float)) and value != 0:
                merged[key] = value
        elif isinstance(current, dict):
            if not current and isinstance(value, dict) and value:
                merged[key] = value
        elif current in (None, '', [], {}):
            if value not in (None, '', [], {}):
                merged[key] = value

    return merged


def _looks_like_noisy_quality(value: str) -> bool:
    if not value:
        return True
    return bool(re.search(r'ROAD|SURAT|GUJARAT|ESTATE|BROKER|PHONE|GSTIN|BLOCK', value, re.IGNORECASE))


def _postprocess_image_challan(challan: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = dict(challan)

    for candidate in candidates:
        quality = str(candidate.get('quality', '')).strip()
        if quality and (_looks_like_noisy_quality(result.get('quality', '')) and not _looks_like_noisy_quality(quality)):
            result['quality'] = quality

        party = str(candidate.get('party', '')).strip()
        if party and re.search(r'GSTIN|CHALLAN|DATE|METER', str(result.get('party', '')), re.IGNORECASE):
            if not re.search(r'GSTIN|CHALLAN|DATE|METER', party, re.IGNORECASE):
                result['party'] = party

        meter = candidate.get('meter', 0.0) or 0.0
        if result.get('meter', 0.0) < 10 and meter > result.get('meter', 0.0):
            result['meter'] = meter

    return result


def clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR text to fix common errors."""
    # Replace common OCR misreadings
    replacements = [
        (r'\(\d+D\)', lambda m: m.group(0).replace('D', '')),  # (8D) -> (8)
        (r'fy\)', ''),            # Noise
        (r'aes\s+Baas', 'as Basic'),  # OCR error pattern
        (r'FagHoIN', 'Fashions'),     # Common OCR error
        (r'JAIMATAD[\)I]', 'JAI MATA DI'),  # OCR error
        (r'(?<=[A-Z])o(?=[A-Z])', '0'),  # Letter O to digit 0 in certain contexts
        # Fix Challan No like "Challan No (8D" -> "Challan No : 8"
        (r'Challan\s+No\s*\((\d+)D?\)?', r'Challan No : \1'),  # (8D or (8D) -> : 8
        # Fix bare 3-digit year like "626" after "Challan Date" -> "2026" (last 2 digits + '20' prefix)
        (r'(Challan\s+Date)\s+(6(\d{2}))\b', r'\1 20\3'),
        # Fix "Qlty :" -> "Qlty:"
        (r'Qlty\s*:', 'Qlty:'),
        # Fix "Meters:" written as "Meter :" etc
        (r'Meter\s*:', 'Meters:'),
    ]

    for pattern, replacement in replacements:
        if callable(replacement):
            text = re.sub(pattern, replacement, text)
        else:
            text = re.sub(pattern, replacement, text)

    return text


def decode_quad(text: str) -> str:
    """Decode quadrupled-font PDF text: JJJJ→J, LLLLLLLL→LL"""
    return re.sub(
        r'(.)\1+',
        lambda m: m.group(1) * (len(m.group(0)) // 4)
                  if len(m.group(0)) % 4 == 0
                  else m.group(0),
        text
    )


def to_float(s: str) -> float:
    try:
        return float(str(s).replace(',', '').strip())
    except (ValueError, TypeError):
        return 0.0


def extract_hsn(text: str) -> str:
    """Extract HSN/SAC code — handles all known PDF variants."""
    patterns = [
        r'HSN\s*/\s*SAC\s*[:\s]+(\d+)',   # HSN / SAC :5407
        r'HSN\s+ACS?\s*[:\s]+(\d+)',        # HSN ACS : 540710
        r'HSN\s+CODE\s*[:\s]+(\d+)',        # HSN CODE 5407
        r'\bHSN\s*[:\s]+(\d+)',             # HSN : 5407
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def extract_all_gstins(text: str) -> List[str]:
    """Extract all unique GSTIN numbers from text (handles special PDF chars and OCR errors)."""
    seen = set()
    result = []
    # More flexible patterns to handle OCR errors
    # Pattern 1: "GST ... : VALUE" or "GST ... No : VALUE"
    colon_matches = re.findall(r'(?:GST|GSTIN)[^:\n]{0,35}:\s*([A-Z0-9]{13,16})', text, re.IGNORECASE)
    # Pattern 2: "GST No space VALUE"
    space_matches = re.findall(r'(?:GST|GSTIN)[\s\-]+([A-Z0-9]{13,16})', text, re.IGNORECASE)
    # Pattern 3: Just look for 13-16 alphanumeric sequences that look like GSTINs
    raw_matches = re.findall(r'\b([0-9]{2}[A-Z0-9]{11,14})\b', text)

    for raw in colon_matches + space_matches + raw_matches:
        cleaned = re.sub(r'[^A-Z0-9]', '', str(raw).upper())
        # Valid GSTIN: 13-15 alphanumeric chars starting with 2 digit state code
        if 13 <= len(cleaned) <= 15 and re.match(r'^\d{2}', cleaned):
            if cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
    return result


def empty_challan() -> Dict[str, Any]:
    return {
        "date": "",
        "challan_date": "",
        "challan_no": "",
        "firm": "JAI MATA DI FASHIONS PVT. LTD.",
        "lot_book_type": "",
        "party": "",
        "party_address": "",
        "master_ac": "",
        "agent": "",
        "gstin_no": "",
        "gstin_map": {},
        "pan_no": "",
        "group": "",
        "marka_help": "",
        "lot_no": "",
        "quality": "",
        "hsn_code": "",
        "taka": 0,
        "meter": 0.0,
        "fas_rate": 0.0,
        "amount": 0.0,
        "dyed_print": "",
        "weight": 0.0,
        "total": 0.0,
        "lr_no": "",
        "lr_date": "",
        "chadhti": 0.0,
        "width": 0.0,
        "transpoter": "",
        "remark": "",
        "weaver": "",
        "item": "",
        "pu_bill_no": "",
        "ms_party": "",
        "delivery_at": "",
        "table": []
    }


def parse_table_format_a(raw_text: str) -> List[Dict]:
    has_weight = bool(re.search(r'TAKA\s+WT', raw_text, re.IGNORECASE))
    table_rows = {}

    for line in raw_text.split('\n'):
        parts = line.strip().split()
        if not parts:
            continue
        try:
            sr1 = int(parts[0])
            if sr1 < 1 or sr1 > 500:
                continue
        except ValueError:
            continue

        nums = re.findall(r'[\d,]+(?:\.\d+)?', line.strip())

        if has_weight:
            # Format: sr mts wt [sr mts wt]
            if len(nums) >= 2:
                try:
                    table_rows[int(nums[0])] = to_float(nums[1])
                except (ValueError, IndexError):
                    pass
            if len(nums) >= 5:
                try:
                    table_rows[int(nums[3])] = to_float(nums[4])
                except (ValueError, IndexError):
                    pass
        else:
            # Format: sr mts [sr mts]
            if len(nums) >= 2:
                try:
                    table_rows[int(nums[0])] = to_float(nums[1])
                except (ValueError, IndexError):
                    pass
            if len(nums) >= 4:
                try:
                    table_rows[int(nums[2])] = to_float(nums[3])
                except (ValueError, IndexError):
                    pass

    return [
        {"srl_no": k, "meter": v}
        for k, v in sorted(table_rows.items())
    ]


def parse_table_format_b(text: str) -> List[Dict]:
    table_rows = {}

    for line in text.split('\n'):
        parts = line.strip().split()
        if not parts:
            continue
        try:
            sr1 = int(parts[0])
            if sr1 < 1 or sr1 > 500:
                continue
        except ValueError:
            continue

        nums = re.findall(r'[\d,]+(?:\.\d+)?', line.strip())

        if len(nums) >= 2:
            try:
                table_rows[int(nums[0])] = to_float(nums[1])
            except (ValueError, IndexError):
                pass
        if len(nums) >= 4:
            try:
                table_rows[int(nums[2])] = to_float(nums[3])
            except (ValueError, IndexError):
                pass

    return [
        {"srl_no": k, "meter": v}
        for k, v in sorted(table_rows.items())
    ]


def parse_format_a(raw_text: str) -> Dict[str, Any]:
    """Parse Job Issue Delivery Challan (quadrupled font — KRISHNA TEXTILES / LUCKY FABRICS style)."""
    c = empty_challan()
    decoded = decode_quad(raw_text)
    lines = [l for l in decoded.split('\n') if l.strip()]

    # Supplier name: 2nd non-empty line (after "Original For Consignee...")
    if len(lines) >= 2:
        candidate = lines[1].strip().strip('*').strip()
        if re.match(r'^[A-Z][A-Z\s&\-\.]+$', candidate):
            c['party'] = candidate

    # Supplier address: line after supplier name, before GSTIN
    if len(lines) >= 3 and not lines[2].strip().upper().startswith('GSTIN'):
        c['party_address'] = lines[2].strip()

    # All GSTINs — search decoded text (handles quadrupled encoding) + raw (catches plain ones)
    all_gstins = extract_all_gstins(decoded) or extract_all_gstins(raw_text)
    c['gstin_no'] = all_gstins[0] if all_gstins else ""

    # PAN No (format: 5 letters + 4 digits + 1 letter = 10 chars)
    pan_m = re.search(r'PAN\s*NO[:\s]+([A-Z]{5}[0-9]{4}[A-Z])', decoded, re.IGNORECASE)
    if pan_m:
        c['pan_no'] = pan_m.group(1).upper()

    # Challan number
    ch_m = re.search(r'CHALLAN\s+NO[.:\s]+([\w/\-]+)', decoded, re.IGNORECASE)
    if ch_m:
        c['challan_no'] = ch_m.group(1)

    # Date (DATE:14/04/26 or DATE:14/04/2026)
    dt_m = re.search(r'DATE[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})', decoded, re.IGNORECASE)
    if dt_m:
        parts = dt_m.group(1).split('/')
        if len(parts) == 3 and len(parts[2]) == 2:
            parts[2] = '20' + parts[2]
        c['date'] = '/'.join(parts)
        c['challan_date'] = c['date']

    # Quality (between QUALITY: and HSN)
    q_m = re.search(r'QUALITY[:\s*]+([A-Z0-9C\s\*\-\.\/]+?)\s+HSN', decoded, re.IGNORECASE)
    if q_m:
        c['quality'] = q_m.group(1).strip().strip('*').strip()

    c['hsn_code'] = extract_hsn(decoded)

    # Marka (e.g. LF-VRUNDAVAN appears as a standalone line before TOTALS)
    mk_m = re.search(r'\n([A-Z]{2,6}-[A-Z]+)\s*\n', decoded)
    if mk_m:
        c['marka_help'] = mk_m.group(1).strip()

    # Total Takas and Total MTS
    tot_m = re.search(
        r'TOTAL\s+TAKAS?\s*[:\s]+(\d+).*?TOTAL\s+MTS[.:\s]+([0-9,.]+)',
        decoded, re.IGNORECASE | re.DOTALL
    )
    if tot_m:
        c['taka'] = int(tot_m.group(1))
        c['meter'] = to_float(tot_m.group(2))

    # Rate and Value (optional — not in all formats)
    rate_m = re.search(r'\bRATE\s*[:\s]+([0-9,.]+)', decoded, re.IGNORECASE)
    if rate_m:
        c['fas_rate'] = to_float(rate_m.group(1))

    val_m = re.search(r'\bVALUE\s*[:\s]+([0-9,.]+)', decoded, re.IGNORECASE)
    if val_m:
        c['amount'] = to_float(val_m.group(1))

    # Weight
    wt_m = re.search(r'\bWEIGHT\s*[:\s]+([0-9,.]+)', decoded, re.IGNORECASE)
    if wt_m:
        c['weight'] = to_float(wt_m.group(1))

    # LR No
    lr_m = re.search(r'LR\s+No\s*[:\s]+([^\n\r]+)', decoded, re.IGNORECASE)
    if lr_m:
        lr_val = lr_m.group(1).strip().strip('-').strip()
        c['lr_no'] = lr_val

    # Sub-party, bill no, pur no → remark (format A specific)
    remark_parts = []
    sp_m = re.search(r'PARTY\s*[:\s]+([A-Z][A-Z\s]+?)(?:\s+PUR|\s*$)', decoded, re.IGNORECASE | re.MULTILINE)
    bl_m = re.search(r'BILL\s+NO[:\s]+(\w+)', decoded, re.IGNORECASE)
    pu_m = re.search(r'PUR\.\s*NO\.\s*[:\s]+(\d+)', decoded, re.IGNORECASE)
    sub_party = sp_m.group(1).strip() if sp_m else ''
    if sub_party:
        remark_parts.append(f"Party: {sub_party}")
    if bl_m:
        remark_parts.append(f"Bill No: {bl_m.group(1)}")
    if pu_m:
        c['pu_bill_no'] = pu_m.group(1)
        remark_parts.append(f"Pur No: {pu_m.group(1)}")
    c['remark'] = ', '.join(remark_parts)

    # Build gstin_map: firm name → GSTIN
    gstin_map = {}
    firm_order = [c['party'], c['firm'], sub_party]
    for i, name in enumerate(firm_order):
        if name and i < len(all_gstins):
            gstin_map[name] = all_gstins[i]
    c['gstin_map'] = gstin_map

    c['table'] = parse_table_format_a(raw_text)
    return c


def parse_table_format_c(text: str) -> List[Dict]:
    """Parse Srl.No. / Taka No. / Meter table (SHREEJI/DHARMIL/HK POLY FAB style)."""
    table_map = {}  # taka_no -> (srl_no, meter)
    in_table = False

    # Header pattern: either "Sr/Srl.No ... Taka No" OR "Taka No ... Sr/Srl.No" (handles OCR column reversal)
    HEADER_RE = re.compile(
        r'(?:'
        r'(?:Srl?\.?\s*No\.?|\bSr\b).*?Taka\s*No'   # normal order
        r'|'
        r'Taka\s*No.*?(?:Srl?\.?\s*No\.?|\bSr\b)'   # reversed order
        r')',
        re.IGNORECASE
    )

    for line in text.split('\n'):
        stripped = line.strip()

        if HEADER_RE.search(stripped):
            in_table = True
            continue

        # Normalize comma-as-decimal (e.g. "120,00" -> "120.00")
        normalized = re.sub(r'(\d+),(\d{2})\b', r'\1.\2', stripped)
        nums = re.findall(r'\d+(?:\.\d+)?', normalized)

        # Auto-enter table mode if we see typical 3-column row data (Srl, Taka, Meter)
        # This is strict to prevent picking up address numbers like '271 TO 273'
        if not in_table and len(nums) >= 3:
            try:
                sr = int(float(nums[0]))
                meter = float(nums[2])
                if 1 <= sr <= 500 and meter > 0:
                    in_table = True
            except ValueError:
                pass

        if not in_table or not nums:
            continue

        if re.search(r'Total\s+Taka', stripped, re.IGNORECASE):
            break

        # Each group is (srl_no, taka_no, meter) — skip if meter == 0
        i = 0
        while i + 2 < len(nums):
            try:
                sr = int(float(nums[i]))
                taka = int(float(nums[i + 1]))
                meter = to_float(nums[i + 2])
                if 1 <= sr <= 500 and taka > 0 and meter > 0 and taka not in table_map:
                    table_map[taka] = (sr, meter)
                i += 3
            except (ValueError, IndexError):
                i += 1

        # Fallback: only 2 numbers on a line -> (srl, meter), no taka column
        if len(nums) == 2:
            try:
                sr = int(float(nums[0]))
                meter = to_float(nums[1])
                if 1 <= sr <= 500 and meter > 0 and sr not in table_map:
                    table_map[sr] = (sr, meter)
            except (ValueError, IndexError):
                pass

        # Single number that looks like a meter reading (>5.0) -> assign to next sr
        elif len(nums) == 1:
            try:
                val = to_float(nums[0])
                if val > 5.0:
                    next_sr = max(table_map.keys(), default=0) + 1
                    if next_sr <= 500 and next_sr not in table_map:
                        table_map[next_sr] = (next_sr, val)
            except (ValueError, IndexError):
                pass

    return sorted(
        [{"srl_no": sr, "tn": taka, "meter": meter} for taka, (sr, meter) in table_map.items()],
        key=lambda x: x['srl_no']
    )


def parse_format_b(raw_text: str) -> Dict[str, Any]:
    """Parse Mill Challan (SUDARSHAN GARMENTS style — plain text, printed twice per page)."""
    c = empty_challan()

    lines = raw_text.split('\n')
    supplier_name = lines[0].strip() if lines else ''

    # Each page prints challan twice — use only first half
    half_idx = len(lines)
    for i in range(10, len(lines)):
        if lines[i].strip() == supplier_name and supplier_name:
            half_idx = i
            break
    text = '\n'.join(lines[:half_idx])

    c['party'] = supplier_name

    # Address lines between name and GSTIN
    addr_parts = []
    for line in lines[1:6]:
        stripped = line.strip()
        if not stripped or re.match(r'GSTIN', stripped, re.IGNORECASE):
            break
        addr_parts.append(stripped)
    c['party_address'] = ' '.join(addr_parts)

    # All GSTINs
    all_gstins = extract_all_gstins(text)
    c['gstin_no'] = all_gstins[0] if all_gstins else ""

    # PAN No
    pan_m = re.search(r'PAN\s*NO[:\s]+([A-Z]{5}[0-9]{4}[A-Z])', text, re.IGNORECASE)
    if pan_m:
        c['pan_no'] = pan_m.group(1).upper()

    # Challan No
    ch_m = re.search(r'Challan\s+No[.:\s]+([\w/\-]+)', text, re.IGNORECASE)
    if ch_m:
        c['challan_no'] = ch_m.group(1)

    # Challan Date
    dt_m = re.search(r'Challan\s+Date\s*[:\s]+(\d{1,2}/\d{1,2}/\d{4})', text, re.IGNORECASE)
    if dt_m:
        c['date'] = dt_m.group(1)
        c['challan_date'] = dt_m.group(1)

    # Item (raw item name, separate from quality)
    item_m = re.search(r'Item\s*[:\s]+(.+?)(?:\s+HSN|\s*$)', text, re.IGNORECASE | re.MULTILINE)
    if item_m:
        c['item'] = item_m.group(1).strip()
        c['quality'] = c['item']

    c['hsn_code'] = extract_hsn(text)

    # Weaver
    wv_m = re.search(r'Weaver\s*[:\s]+([^\n]+)', text, re.IGNORECASE)
    if wv_m:
        c['weaver'] = wv_m.group(1).strip()

    # Pu.BillNo
    pu_m = re.search(r'Pu\.?\s*BillNo\s*[:\s]+(\S+)', text, re.IGNORECASE)
    if pu_m:
        c['pu_bill_no'] = pu_m.group(1).strip()

    # Total Taka
    tt_m = re.search(r'Total\s+Taka\s*[:\s]+(\d+)', text, re.IGNORECASE)
    if tt_m:
        c['taka'] = int(tt_m.group(1))

    # Total Mts
    tm_m = re.search(r'Total\s+Mts[.:\s]+([0-9,.]+)', text, re.IGNORECASE)
    if tm_m:
        c['meter'] = to_float(tm_m.group(1))

    # Build gstin_map: party → gstin[0], firm (consignee) → gstin[1]
    gstin_map = {}
    if c['party'] and all_gstins:
        gstin_map[c['party']] = all_gstins[0]
    if c['firm'] and len(all_gstins) > 1:
        gstin_map[c['firm']] = all_gstins[1]
    c['gstin_map'] = gstin_map

    c['table'] = parse_table_format_b(text)
    return c


def parse_format_c(raw_text: str, layout_text: str = '') -> Dict[str, Any]:
    """Parse Delivery Challan (DHARMIL TEXTILES / SHREEJI style — Srl.No./Taka No./Meter table)."""
    c = empty_challan()
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

    c['party'] = lines[0] if lines else ''
    party_index = 0
    # Skip leading OCR noise lines and obvious headers.
    skip_party_patterns = (
        r'GSTIN|DELIVERY\s+CHALLAN|CHALLAN\s+NO|DATE|PHONE|BROC?KER|M/S|TOTAL|SRI?L?\.?\s*NO'
    )
    for idx, line in enumerate(lines):
        normalized_line = re.sub(r'[^A-Z ]', '', line.upper()).strip()
        if (
            len(line) >= 3
            and re.search(r'[A-Z]', line)
            and not re.search(skip_party_patterns, line, re.IGNORECASE)
            and len(normalized_line) >= 6
        ):
            c['party'] = line
            party_index = idx
            break

    addr_parts = []
    for line in lines[party_index + 1:party_index + 5]:
        if re.match(r'(Phone|GSTIN)', line, re.IGNORECASE):
            break
        addr_parts.append(line)
    c['party_address'] = ' '.join(addr_parts)

    all_gstins = extract_all_gstins(raw_text)
    c['gstin_no'] = all_gstins[0] if all_gstins else ''

    # More flexible challan number pattern - handles OCR errors like "(8D)" instead of ": 3"
    ch_patterns = [
        r'Challan\s+No[.:\s]+(\d+)',
        r'Challan\s+No[:\s]*\(?\s*(\d+)',   # Handles OCR errors like "No(8" or "No(3"
        r'Challan\s+[Nn]o[^0-9]*(\d{1,4})', # Generic: Challan No followed by 1-4 digits
    ]
    for pat in ch_patterns:
        ch_m = re.search(pat, raw_text, re.IGNORECASE)
        if ch_m:
            num = ch_m.group(1)
            # Take the first 1-4 digits to avoid OCR corruption
            num_match = re.match(r'(\d{1,4})', num)
            if num_match:
                num = num_match.group(1)
                if int(num) < 10000:  # Reasonable challan number range
                    c['challan_no'] = num
            break

    # More flexible date patterns - handles partial dates from OCR
    dt_patterns = [
        r'(?:Challan\s+)?Date\s*[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        r'Date\s*[:\s]+(\d{4})',   # Just year (e.g., "2026")
        r'Date\s*[:\s]+(20\d{2})', # Explicit 20XX year
    ]
    for pat in dt_patterns:
        dt_m = re.search(pat, raw_text, re.IGNORECASE)
        if dt_m:
            date_val = dt_m.group(1).replace('-', '/')
            if '/' in date_val or (len(date_val) == 4 and date_val.isdigit() and 2000 <= int(date_val) <= 2100):
                c['date'] = date_val
                c['challan_date'] = date_val
            break

    # Fallback: Try to find bare 3 or 4-digit year after "Challan Date"
    if not c['date']:
        year_m = re.search(r'Challan\s+Date\s+(\d{3,4})', raw_text, re.IGNORECASE)
        if year_m:
            year_str = year_m.group(1)
            # 3-digit OCR year e.g. "626" -> last 2 digits = "26" -> "2026"
            if len(year_str) == 3:
                year_str = '20' + year_str[-2:]
            if year_str.isdigit() and 2000 <= int(year_str) <= 2100:
                c['date'] = year_str
                c['challan_date'] = year_str

    # Match Qlty, Qlty:, Quality:, Quality *: (handles OCR spacing)
    q_m = re.search(r'Ql(?:ty|uality)\s*[:\*\s]+([^\n]+?)\s*(?:\n|$)', raw_text, re.IGNORECASE)
    if not q_m:
        q_m = re.search(r'Quality\s*[:\s]+([^\n]+)', raw_text, re.IGNORECASE)
    if q_m:
        c['quality'] = q_m.group(1).strip()

    c['hsn_code'] = extract_hsn(raw_text)

    broker_m = re.search(r'Broc?ker\s*[:\s]+([^\n]+)', raw_text, re.IGNORECASE)
    if broker_m:
        c['agent'] = broker_m.group(1).strip()

    tt_m = re.search(r'Total\s+Taka\s*[:\s]+(\d+)', raw_text, re.IGNORECASE)
    if tt_m:
        c['taka'] = int(tt_m.group(1))

    # More flexible meter pattern - handles "Meters: 1611.00", "Mts: 1611", "Total Taka: 14 Meters: 1611.00"
    # First try combined pattern (most specific)
    tm_m = re.search(r'Total\s+(?:Taka|Mts)[^\n]*?Meters?\s*[:\s.]+([0-9,.]+)', raw_text, re.IGNORECASE)
    if tm_m:
        c['meter'] = to_float(tm_m.group(1))
    else:
        # Try standalone Meters: / Mts: patterns
        for pat in [
            r'Total\s+Meters?\s*[:\s._-]+([0-9,.]+)',
            r'Total\s+Mts\.?\s*[:\s]+([0-9,.]+)',
        ]:
            tm_m = re.search(pat, raw_text, re.IGNORECASE)
            if tm_m:
                c['meter'] = to_float(tm_m.group(1))
                break

    # Use layout text (two-column aware) for M/s party, Delivery At, and Remark
    if layout_text:
        layout_lines = layout_text.split('\n')
        for i, line in enumerate(layout_lines):
            if re.search(r'Party\.?\s+Delivery\s+At', line, re.IGNORECASE):
                data_lines = []
                for candidate_line in layout_lines[i + 1:i + 4]:
                    if re.search(r'Broc?ker|GSTIN|Remark|Sri\.?No|Taka\s*No', candidate_line, re.IGNORECASE):
                        break
                    data_lines.append(candidate_line)
                data_line = re.sub(r'\s+Date\s*:?.*$', '', ' '.join(data_lines))
                parts = [p.strip() for p in re.split(r'\s{3,}', data_line) if p.strip()]
                c['ms_party'] = parts[0] if parts else ''
                c['delivery_at'] = parts[1] if len(parts) > 1 else ''
                break

        for line in layout_lines:
            if re.search(r'Remark\s*[:.]', line, re.IGNORECASE):
                remark_m = re.search(r'Remark\s*[:.]\s*(.+)', line)
                if remark_m:
                    c['remark'] = remark_m.group(1).strip(' .')
                break

    if not c['ms_party'] or not c['delivery_at']:
        combined_layout = re.sub(r'\s+', ' ', layout_text or raw_text)
        delivery_match = re.search(
            r'(J[AI/\\\s]*MATA\s*D[I1]?\s*FASHIONS?\s*PVT\.?\s*LTD\.?(?:\s*\(UNIT-?\d+\))?)',
            combined_layout,
            re.IGNORECASE
        )
        if delivery_match:
            c['delivery_at'] = c['delivery_at'] or delivery_match.group(1).strip(' |:-')
            prefix = combined_layout[:delivery_match.start()]
            prefix = re.sub(r'.*?(?:M/s\.?|Party\.?\s*Delivery\s*At[: ]?)', '', prefix, flags=re.IGNORECASE)
            prefix = re.sub(r'\b(?:Challan|Date|Quality|Broker|GSTIN|Tempo)\b.*$', '', prefix, flags=re.IGNORECASE)
            prefix = prefix.strip(' |:-')
            if prefix:
                c['ms_party'] = c['ms_party'] or prefix

        party_block = re.search(
            r'(?:M/s\.?\s*[: ]\s*|M/s\.?\s*Party\.?\s*Delivery\s*At[: ]\s*)'
            r'(.+?)\s+(JAI\s*MATA\s*D[I1]\s*FASHIONS?\s*PVT\.?\s*LTD\.?(?:\s*\(UNIT-?\d+\))?)'
            r'(?=\s+(?:Challan|Date|Quality|Broker|GSTIN|Tempo)\b)',
            combined_layout,
            re.IGNORECASE
        )
        if party_block:
            c['ms_party'] = c['ms_party'] or party_block.group(1).strip(' |:-')
            c['delivery_at'] = c['delivery_at'] or party_block.group(2).strip(' |:-')

    # Build gstin_map after ms_party/delivery_at are set by layout parsing above
    gstin_map = {}
    firm_order = [c['party'], c['ms_party'], c['delivery_at']]
    for i, name in enumerate(firm_order):
        if name and i < len(all_gstins):
            gstin_map[name] = all_gstins[i]
    c['gstin_map'] = gstin_map

    c['table'] = parse_table_format_c(raw_text)
    return c


def detect_format(text: str) -> str:
    if re.search(r'JJJJ|DDDDEEEELLLLIIIIVVVVEEEERRRRYYYY', text):
        return 'A'
    if re.search(r'MILL\s+CHALLAN', text, re.IGNORECASE):
        return 'B'
    # Format C: relaxed — handles garbled OCR headers like "Sr TakaNo", "Srl.No. Taka No rs"
    if re.search(r'(?:Srl?\.?\s*No\.?|\bSr\b).*?Taka\s*No', text, re.IGNORECASE):
        return 'C'
    # Also treat plain DELIVERY CHALLAN docs as format C
    if re.search(r'DELIVERY\s+CHALLAN', text, re.IGNORECASE):
        return 'C'
    return 'A'


def extract_challans_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    challans = []
    seen = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ''
            if not text.strip():
                continue

            fmt = detect_format(text)
            if fmt == 'A':
                challan = parse_format_a(text)
            elif fmt == 'B':
                challan = parse_format_b(text)
            else:
                layout_text = page.extract_text(layout=True) or ''
                challan = parse_format_c(text, layout_text)

            cn = challan.get('challan_no', '')
            if cn and cn in seen:
                continue
            if cn:
                seen.add(cn)

            challans.append(challan)

    return challans


def extract_challans(file_path: str) -> List[Dict[str, Any]]:
    """Extract challans from PDF or image (JPEG/PNG) files."""
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        return extract_challans_from_pdf(file_path)
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        default_text = clean_ocr_text(extract_text_from_image(file_path))
        enhanced_text = clean_ocr_text(
            extract_text_from_image(file_path, preprocess=True, config=r'--oem 3 --psm 6')
        )
        layout_text = clean_ocr_text(extract_layout_text_from_image(file_path))
        table_text = clean_ocr_text(extract_table_text_from_image(file_path))

        text_candidates = [text for text in [default_text, enhanced_text, layout_text, table_text] if text.strip()]
        if not text_candidates:
            return []

        merged_text = '\n'.join(dict.fromkeys(text_candidates))
        fmt = detect_format(merged_text)

        parsed_candidates = []
        for text in dict.fromkeys(text_candidates + [merged_text]):
            if fmt == 'A':
                parsed = parse_format_a(text)
            elif fmt == 'B':
                parsed = parse_format_b(text)
            else:
                parsed = parse_format_c(text, layout_text=layout_text or text)
            parsed_candidates.append(parsed)

        challan = parsed_candidates[0]
        for candidate in parsed_candidates[1:]:
            challan = _merge_extracted_challans(challan, candidate)
        challan = _postprocess_image_challan(challan, parsed_candidates)

        if table_text:
            if fmt == 'A':
                challan['table'] = parse_table_format_a(table_text)
            elif fmt == 'B':
                challan['table'] = parse_table_format_b(table_text)
            else:
                parsed_table = parse_table_format_c(table_text)
                if len(parsed_table) > len(challan.get('table', [])):
                    challan['table'] = parsed_table

        return [challan]
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported types: .pdf, .jpg, .jpeg, .png")
