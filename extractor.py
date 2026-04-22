import re
import pdfplumber
from typing import List, Dict, Any


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


def extract_all_gstins(text: str) -> List[str]:
    """Extract all unique GSTIN numbers from text (handles special PDF chars)."""
    seen = set()
    result = []
    for raw in re.findall(r'GSTIN[:\s\-]+(\S{14,16})', text, re.IGNORECASE):
        cleaned = re.sub(r'[^A-Z0-9]', '', raw.upper())
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
        "gstin_numbers": [],
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
        {"tn": k, "meter": v, "finish_mtr": 0.0, "shortage_mtr": 0.0, "ba_mtr": 0.0}
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
        {"tn": k, "meter": v, "finish_mtr": 0.0, "shortage_mtr": 0.0, "ba_mtr": 0.0}
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
    c['gstin_numbers'] = all_gstins
    c['gstin_no'] = all_gstins[0] if all_gstins else ""

    # PAN No (format: 5 letters + 4 digits + 1 letter = 10 chars)
    pan_m = re.search(r'PAN\s*NO[:\s]+([A-Z]{5}[0-9]{4}[A-Z])', decoded, re.IGNORECASE)
    if pan_m:
        c['pan_no'] = pan_m.group(1).upper()

    # Challan number
    ch_m = re.search(r'CHALLAN\s+NO[.:\s]+(\d+)', decoded, re.IGNORECASE)
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

    # HSN
    hsn_m = re.search(r'HSN\s*/?\s*SAC\s*[:\s]+(\d+)', decoded, re.IGNORECASE)
    if hsn_m:
        c['hsn_code'] = hsn_m.group(1)

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

    # Sub-party, bill no, pur no → remark
    remark_parts = []
    sp_m = re.search(r'PARTY\s*[:\s]+([A-Z][A-Z\s]+?)(?:\s+PUR|\s*$)', decoded, re.IGNORECASE | re.MULTILINE)
    bl_m = re.search(r'BILL\s+NO[:\s]+(\w+)', decoded, re.IGNORECASE)
    pu_m = re.search(r'PUR\.\s*NO\.\s*[:\s]+(\d+)', decoded, re.IGNORECASE)
    if sp_m:
        remark_parts.append(f"Party: {sp_m.group(1).strip()}")
    if bl_m:
        remark_parts.append(f"Bill No: {bl_m.group(1)}")
    if pu_m:
        remark_parts.append(f"Pur No: {pu_m.group(1)}")
    c['remark'] = ', '.join(remark_parts)

    c['table'] = parse_table_format_a(raw_text)
    return c


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
    c['gstin_numbers'] = all_gstins
    c['gstin_no'] = all_gstins[0] if all_gstins else ""

    # PAN No
    pan_m = re.search(r'PAN\s*NO[:\s]+([A-Z]{5}[0-9]{4}[A-Z])', text, re.IGNORECASE)
    if pan_m:
        c['pan_no'] = pan_m.group(1).upper()

    # Challan No
    ch_m = re.search(r'Challan\s+No[.:\s]+(\d+)', text, re.IGNORECASE)
    if ch_m:
        c['challan_no'] = ch_m.group(1)

    # Challan Date
    dt_m = re.search(r'Challan\s+Date\s*[:\s]+(\d{1,2}/\d{1,2}/\d{4})', text, re.IGNORECASE)
    if dt_m:
        c['date'] = dt_m.group(1)
        c['challan_date'] = dt_m.group(1)

    # Item / Quality (up to HSN or end of line)
    item_m = re.search(r'Item\s*[:\s]+(.+?)(?:\s+HSN|\s*$)', text, re.IGNORECASE | re.MULTILINE)
    if item_m:
        c['quality'] = item_m.group(1).strip()

    # HSN
    hsn_m = re.search(r'HSN\s+ACS?\s*[:\s]+(\d+)', text, re.IGNORECASE)
    if hsn_m:
        c['hsn_code'] = hsn_m.group(1)

    # Weaver and Pu.BillNo → remark
    remark_parts = []
    wv_m = re.search(r'Weaver\s*[:\s]+([^\n]+)', text, re.IGNORECASE)
    pu_m = re.search(r'Pu\.?\s*BillNo\s*[:\s]+(\S+)', text, re.IGNORECASE)
    if wv_m:
        remark_parts.append(f"Weaver: {wv_m.group(1).strip()}")
    if pu_m:
        remark_parts.append(f"Pu.BillNo: {pu_m.group(1).strip()}")
    c['remark'] = ', '.join(remark_parts)

    # Total Taka
    tt_m = re.search(r'Total\s+Taka\s*[:\s]+(\d+)', text, re.IGNORECASE)
    if tt_m:
        c['taka'] = int(tt_m.group(1))

    # Total Mts
    tm_m = re.search(r'Total\s+Mts[.:\s]+([0-9,.]+)', text, re.IGNORECASE)
    if tm_m:
        c['meter'] = to_float(tm_m.group(1))

    c['table'] = parse_table_format_b(text)
    return c


def detect_format(text: str) -> str:
    if re.search(r'JJJJ|DDDDEEEELLLLIIIIVVVVEEEERRRRYYYY', text):
        return 'A'
    if re.search(r'MILL\s+CHALLAN', text, re.IGNORECASE):
        return 'B'
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
            challan = parse_format_a(text) if fmt == 'A' else parse_format_b(text)

            cn = challan.get('challan_no', '')
            if cn and cn in seen:
                continue
            if cn:
                seen.add(cn)

            challans.append(challan)

    return challans
