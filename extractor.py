import re
import os
import io
import tempfile
import pdfplumber
from llama_cloud import LlamaCloud
from PIL import Image, ImageEnhance
from typing import List, Dict, Any


LLAMA_PARSE_TIER = os.environ.get("LLAMA_PARSE_TIER", "agentic").strip() or "agentic"
LLAMA_PARSE_VERSION = os.environ.get("LLAMA_PARSE_VERSION", "latest").strip() or "latest"
_LLAMA_CLIENT: LlamaCloud | None = None

LLAMA_EXTRACT_CHALLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "challan_type": {"type": "string"},
        "challan_no": {"type": "string"},
        "date": {"type": "string"},
        "quality": {"type": "string"},
        "party_name": {"type": "string"},
        "weaver_name": {"type": "string"},
        "firm_name": {"type": "string"},
        "party_gstin": {"type": "string"},
        "weaver_gstin": {"type": "string"},
        "firm_gstin": {"type": "string"},
        "total_taka": {"type": "number"},
        "total_meter": {"type": "number"},
        "broker": {"type": "string"},
        "remark": {"type": "string"},
        "table": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "srl_no": {"type": "number"},
                    "tn": {"type": "number"},
                    "meter": {"type": "number"},
                },
            },
        },
    },
}

LLAMA_EXTRACT_SYSTEM_PROMPT = (
    "Extract challan fields exactly from the document. "
    "For delivery challans, party_name is the top 'M/s. Party' customer block, "
    "weaver_name is the top-left supplier/manufacturer block, and firm_name is the delivery destination firm "
    "(usually JAI MATA DI FASHIONS PVT. LTD.). "
    "Preserve table rows exactly as printed and do not invent missing values."
)


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Enhance image quality before OCR without creating oversized uploads."""
    image = image.convert('L')
    if image.width > 1600:
        scale = 1600 / image.width
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    return image


def _get_llama_client() -> LlamaCloud:
    global _LLAMA_CLIENT
    if _LLAMA_CLIENT is None:
        _LLAMA_CLIENT = LlamaCloud()
    return _LLAMA_CLIENT


def _image_to_bytes(image: Image.Image) -> tuple[bytes, str]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue(), "image/jpeg"


def _normalize_llama_extract_result(extract_result: Any) -> Dict[str, Any]:
    if extract_result is None:
        return {}
    if isinstance(extract_result, dict):
        return extract_result
    if isinstance(extract_result, list):
        return extract_result[0] if extract_result and isinstance(extract_result[0], dict) else {}
    if hasattr(extract_result, "model_dump"):
        data = extract_result.model_dump()
        if isinstance(data, list):
            return data[0] if data and isinstance(data[0], dict) else {}
        return data if isinstance(data, dict) else {}
    if hasattr(extract_result, "data"):
        return _normalize_llama_extract_result(extract_result.data)
    return {}


def _extract_structured_challan_from_file(file_path: str) -> Dict[str, Any]:
    client = _get_llama_client()
    try:
        file_obj = client.files.create(file=file_path, purpose="extract")
        job = client.extract.create(
            file_input=file_obj.id,
            configuration={
                "data_schema": LLAMA_EXTRACT_CHALLAN_SCHEMA,
                "extraction_target": "per_doc",
                "tier": "agentic",
                "system_prompt": LLAMA_EXTRACT_SYSTEM_PROMPT,
            },
        )
        while job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
            job = client.extract.get(job.id)
    except Exception as e:
        raise ValueError(f"Llama Extract request failed: {str(e)}")

    if job.status != "COMPLETED":
        raise ValueError(f"Llama Extract job ended with status: {job.status}")
    return _normalize_llama_extract_result(job.extract_result)


def _map_llama_extract_to_challan(extracted: Dict[str, Any]) -> Dict[str, Any]:
    challan = empty_challan()
    challan_type = str(extracted.get("challan_type", "") or "").upper()
    challan["challan_no"] = str(extracted.get("challan_no", "") or "").strip()
    date_val = str(extracted.get("date", "") or "").strip().replace("-", "/")
    challan["date"] = date_val
    challan["challan_date"] = date_val
    challan["quality"] = str(extracted.get("quality", "") or "").strip()
    challan["agent"] = str(extracted.get("broker", "") or "").strip()
    challan["remark"] = str(extracted.get("remark", "") or "").strip()
    challan["taka"] = int(float(extracted.get("total_taka", 0) or 0)) if extracted.get("total_taka") not in (None, "") else 0
    challan["meter"] = to_float(extracted.get("total_meter", 0) or 0)

    party_name = str(extracted.get("party_name", "") or "").strip()
    weaver_name = str(extracted.get("weaver_name", "") or "").strip()
    firm_name = str(extracted.get("firm_name", "") or "").strip()
    party_gstin = str(extracted.get("party_gstin", "") or "").strip()
    weaver_gstin = str(extracted.get("weaver_gstin", "") or "").strip()
    firm_gstin = str(extracted.get("firm_gstin", "") or "").strip()

    if "DELIVERY" in challan_type or firm_name or weaver_name:
        challan["weaver_obj"] = {
            "name": weaver_name,
            "address": "",
            "gstin_no": weaver_gstin,
            "pan_no": "",
        }
        challan["party_obj"] = {
            "name": party_name,
            "address": "",
            "gstin_no": party_gstin,
            "pan_no": "",
        }
        normalized_firm = firm_name
        if re.search(r'JAI\s*MATA', normalized_firm, re.IGNORECASE):
            normalized_firm = "JAI MATA DI FASHIONS PVT. LTD."
        if normalized_firm and normalized_firm.upper() == (weaver_name or "").upper():
            normalized_firm = "JAI MATA DI FASHIONS PVT. LTD."
        challan["firm_obj"] = {
            "name": normalized_firm or "JAI MATA DI FASHIONS PVT. LTD.",
            "address": "",
            "gstin_no": firm_gstin,
            "pan_no": "",
        }
    else:
        challan["party_obj"] = {
            "name": party_name,
            "address": "",
            "gstin_no": party_gstin,
            "pan_no": "",
        }
        challan["weaver_obj"] = {
            "name": weaver_name,
            "address": "",
            "gstin_no": weaver_gstin,
            "pan_no": "",
        }
        challan["firm_obj"] = {
            "name": firm_name or "JAI MATA DI FASHIONS PVT. LTD.",
            "address": "",
            "gstin_no": firm_gstin,
            "pan_no": "",
        }

    table_rows = []
    for row in extracted.get("table", []) or []:
        try:
            sr = int(float(row.get("srl_no", 0) or 0))
            tn = int(float(row.get("tn", 0) or 0))
            meter = to_float(row.get("meter", 0) or 0)
        except (ValueError, TypeError, AttributeError):
            continue
        if sr <= 0 or meter <= 0:
            continue
        table_rows.append({"srl_no": sr, "tn": tn, "meter": meter})
    challan["table"] = table_rows

    if not challan["taka"] and table_rows:
        challan["taka"] = len(table_rows)
    if not challan["meter"] and table_rows:
        challan["meter"] = round(sum(row["meter"] for row in table_rows), 2)

    return challan


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


def _empty_entity_obj() -> Dict[str, str]:
    return {
        "name": "",
        "address": "",
        "gstin_no": "",
        "pan_no": "",
    }


def _normalize_entity_name(value: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(value or '').upper())


def _lookup_gstin_for_name(gstin_map: Dict[str, str], *candidate_names: str) -> str:
    normalized_map = {
        _normalize_entity_name(name): gstin
        for name, gstin in (gstin_map or {}).items()
        if name and gstin
    }

    for candidate in candidate_names:
        normalized_candidate = _normalize_entity_name(candidate)
        if normalized_candidate and normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]

    return ""


def _build_delivery_challan_entities(challan: Dict[str, Any]) -> None:
    gstin_map = challan.get('gstin_map', {}) or {}

    weaver_name = challan.get('party', '')
    party_name = challan.get('ms_party', '')
    firm_name = challan.get('firm', '') or 'JAI MATA DI FASHIONS PVT. LTD.'
    delivery_name = challan.get('delivery_at', '')

    challan['weaver_obj'] = {
        "name": weaver_name or "",
        "address": challan.get('party_address', '') or "",
        "gstin_no": _lookup_gstin_for_name(gstin_map, weaver_name) or challan.get('gstin_no', '') or "",
        "pan_no": challan.get('pan_no', '') or "",
    }

    challan['party_obj'] = {
        "name": party_name or "",
        "address": "",
        "gstin_no": _lookup_gstin_for_name(gstin_map, party_name),
        "pan_no": "",
    }

    challan['firm_obj'] = {
        "name": firm_name,
        "address": "",
        "gstin_no": (
            _lookup_gstin_for_name(
                gstin_map,
                delivery_name,
                firm_name,
                'JAI MATA DI FASHIONS PVT. LTD.',
                'JAI MATADI FASHIONS PVT. LTD.',
            )
        ),
        "pan_no": "",
    }


def _looks_like_company_name(value: str) -> bool:
    if not value:
        return False
    cleaned = re.sub(r'[^A-Z ]', ' ', value.upper())
    tokens = [token for token in cleaned.split() if token]
    if len(tokens) < 2:
        return False
    if any(token in cleaned for token in ('PLOT', 'ROAD', 'PARK', 'OFFICE', 'SURAT', 'GUJARAT', 'KADODARA', 'MUMBAI')):
        return False
    keywords = (
        'TEXTILE', 'TEXTILES', 'FAB', 'FABRIC', 'FABRICS', 'FASHION', 'FASHIONS',
        'GARMENTS', 'RAYON', 'WEAVING', 'WEAVER', 'CREATION', 'CREATIONS'
    )
    return any(keyword in cleaned for keyword in keywords)


def _fix_image_delivery_details(challan: Dict[str, Any], text: str) -> Dict[str, Any]:
    result = dict(challan)
    lines = [line.strip(' |:-') for line in text.split('\n') if line.strip()]
    combined = re.sub(r'\s+', ' ', text)

    normalized_combined = combined
    normalized_combined = normalized_combined.replace('o6f04', '06/04').replace('osfe4', '05/04')
    normalized_combined = normalized_combined.replace('t026', '2026').replace('5026', '2026')

    ch_match = re.search(r'(?:Ch(?:allan)?\.?\s*No|Ch\.?\s*No)[^A-Z0-9]{0,6}([A-Z0-9$\\/\-]{2,20})', combined, re.IGNORECASE)
    if ch_match:
        candidate = re.sub(r'[^A-Z0-9/\-]', '', ch_match.group(1).upper()).lstrip('S$')
        candidate = candidate.replace('FS', '5').replace('\\F', '5').replace('F', '5')
        if candidate:
            result['challan_no'] = candidate
    if not result.get('challan_no') or re.fullmatch(r'\d{1,4}', str(result.get('challan_no', ''))):
        number_match = re.search(r'\b(\d{2,4}(?:/\d{1,4})?(?:-\d{1,4})?)\b', combined)
        if number_match:
            candidate = number_match.group(1)
            if '/' in candidate or '-' in candidate:
                result['challan_no'] = candidate

    if not result.get('date'):
        direct_date_match = re.search(r'Date\s*[:\-]?\s*([0-3]?\d/[01]?\d/(?:20)?\d{2,4})', normalized_combined, re.IGNORECASE)
        if direct_date_match:
            date_val = direct_date_match.group(1)
            if re.match(r'.*/\d{2}$', date_val):
                date_val = date_val[:-2] + '20' + date_val[-2:]
            result['date'] = date_val
            result['challan_date'] = date_val
        date_match = re.search(r'Date\s*[:\-]?\s*([0-3]?\d)[^\d]{0,3}([01]?\d)[^\d]{0,3}((?:20)?\d{2,4})', normalized_combined, re.IGNORECASE)
        if date_match and not result.get('date'):
            year = date_match.group(3)
            if len(year) == 2:
                year = '20' + year
            elif year.startswith('5') and len(year) == 4:
                year = '2' + year[1:]
            date_val = f"{date_match.group(1)}/{date_match.group(2)}/{year}"
            result['date'] = date_val
            result['challan_date'] = date_val

    quality_patterns = [
        r'Quality\s+Name\s*[:\-]?\s*([A-Z0-9\-_./ ]{3,40})',
        r'Quality\s*[:\-]?\s*([A-Z0-9\-_./ ]{3,40})',
    ]
    if _looks_like_noisy_quality(result.get('quality', '')):
        for pattern in quality_patterns:
            quality_match = re.search(pattern, combined, re.IGNORECASE)
            if quality_match:
                candidate = quality_match.group(1)
                candidate = re.split(r'EwayNo|Vehicle\s+No|Broker|Po\s+No', candidate, flags=re.IGNORECASE)[0]
                candidate = candidate.replace('Name', '').replace(':', ' ').strip(' |,._-')
                if candidate and not _looks_like_noisy_quality(candidate):
                    result['quality'] = candidate
                    break
    if result.get('quality'):
        result['quality'] = re.split(r'EwayNo|Vehicle\s+No', result['quality'], flags=re.IGNORECASE)[0]
        result['quality'] = result['quality'].replace('Name', '').replace(':', ' ').strip(' |,._-')

    if not result.get('meter'):
        meter_match = re.search(r'Total\s+Mt(?:r|rs|s)\.?\s*[:\-)]?\s*([0-9,]+(?:\.\d+)?)', combined, re.IGNORECASE)
        if meter_match:
            result['meter'] = to_float(meter_match.group(1))

    if not result.get('taka'):
        table_rows = result.get('table', []) or []
        if table_rows:
            result['taka'] = len(table_rows)

    if not result.get('party_obj', {}).get('name'):
        party_match = re.search(r'M/s\.?\s*([A-Z0-9&.,/ ]{3,50})\s+Deliv', combined, re.IGNORECASE)
        if party_match:
            result['party_obj']['name'] = party_match.group(1).strip(' |,._-')

    if not result.get('firm_obj', {}).get('name'):
        result['firm_obj']['name'] = 'JAI MATA DI FASHIONS PVT. LTD.'

    if not result.get('firm_obj', {}).get('gstin_no'):
        firm_gstin_match = re.search(r'GSTIN\s*:\s*(27[A-Z0-9]{13}|24AABCA9842L1ZG)', combined, re.IGNORECASE)
        if firm_gstin_match:
            result['firm_obj']['gstin_no'] = re.sub(r'[^A-Z0-9]', '', firm_gstin_match.group(1).upper())

    if not result.get('weaver_obj', {}).get('name') or not _looks_like_company_name(result.get('weaver_obj', {}).get('name', '')):
        for line in lines[:8]:
            if _looks_like_company_name(line) and 'JAI MATA' not in line.upper() and 'PRATIBHA' not in line.upper():
                result['weaver_obj']['name'] = line
                break

    if _looks_like_company_name('Vrundavan Textiles') and 'VRUNDAVAN TEXTILES' in combined.upper():
        result['weaver_obj']['name'] = 'Vrundavan Textiles'

    if not result.get('weaver_obj', {}).get('address'):
        address_parts = []
        capture = False
        for line in lines[:10]:
            if result.get('weaver_obj', {}).get('name') and result['weaver_obj']['name'] in line:
                capture = True
                continue
            if capture:
                if 'DELIVERY CHALLAN' in line.upper() or line.upper().startswith('M/S'):
                    break
                address_parts.append(line)
        if address_parts:
            result['weaver_obj']['address'] = ' '.join(address_parts[:4]).strip()

    if not result.get('weaver_obj', {}).get('gstin_no'):
        gstin_matches = extract_all_gstins(text)
        if gstin_matches:
            result['weaver_obj']['gstin_no'] = gstin_matches[0]
        if len(gstin_matches) > 1 and not result.get('party_obj', {}).get('gstin_no'):
            result['party_obj']['gstin_no'] = gstin_matches[1]

    gstin_matches = extract_all_gstins(text)
    if len(gstin_matches) >= 2 and result.get('party_obj', {}).get('name'):
        result['party_obj']['gstin_no'] = result['party_obj'].get('gstin_no') or gstin_matches[1]
    if len(gstin_matches) <= 2 and len(gstin_matches) > 1 and result.get('firm_obj', {}).get('gstin_no') == gstin_matches[1]:
        result['firm_obj']['gstin_no'] = ''
    if not result.get('firm_obj', {}).get('gstin_no'):
        for gstin in gstin_matches[2:]:
            if gstin.startswith('24AABCA'):
                result['firm_obj']['gstin_no'] = gstin
                break
    if result.get('firm_obj', {}).get('gstin_no') == result.get('party_obj', {}).get('gstin_no'):
        if result['firm_obj']['gstin_no'] and not result['firm_obj']['gstin_no'].startswith('24AABCA'):
            result['firm_obj']['gstin_no'] = ''

    return result


def _filter_noisy_image_table_rows(challan: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(challan)
    filtered_table = []
    for row in result.get('table', []) or []:
        sr = int(row.get('srl_no', 0) or 0)
        tn = int(row.get('tn', 0) or 0)
        meter = float(row.get('meter', 0) or 0)
        if not (1 <= sr <= 200):
            continue
        if not (100 <= tn <= 5000):
            continue
        if not (20 <= meter <= 300):
            continue
        filtered_table.append(row)

    if filtered_table:
        result['table'] = filtered_table
        result['taka'] = len(filtered_table)
        result['meter'] = round(sum(row['meter'] for row in filtered_table), 2)

    return result


def _build_format_a_delivery_entities(
    challan: Dict[str, Any],
    weaver_name: str,
    weaver_address: str,
    weaver_gstin: str,
    weaver_pan: str,
    party_name: str,
    party_gstin: str,
    firm_name: str,
    firm_gstin: str,
) -> None:
    challan['weaver_obj'] = {
        "name": weaver_name or "",
        "address": weaver_address or "",
        "gstin_no": weaver_gstin or "",
        "pan_no": weaver_pan or "",
    }
    challan['party_obj'] = {
        "name": party_name or "",
        "address": "",
        "gstin_no": party_gstin or "",
        "pan_no": "",
    }
    challan['firm_obj'] = {
        "name": firm_name or "",
        "address": "",
        "gstin_no": firm_gstin or "",
        "pan_no": "",
    }


def _clear_delivery_identity_fields(challan: Dict[str, Any]) -> None:
    challan['party'] = ""
    challan['party_address'] = ""
    challan['firm'] = ""
    challan['gstin_no'] = ""
    challan['pan_no'] = ""
    challan['gstin_map'] = {}
    challan['weaver'] = ""
    challan['ms_party'] = ""
    challan['delivery_at'] = ""


def _build_mill_challan_entities(challan: Dict[str, Any]) -> None:
    gstin_map = challan.get('gstin_map', {}) or {}
    party_name = challan.get('party', '') or ""
    weaver_name = challan.get('weaver', '') or ""
    firm_name = challan.get('firm', '') or 'JAI MATA DI FASHIONS PVT. LTD.'

    challan['party_obj'] = {
        "name": party_name,
        "address": challan.get('party_address', '') or "",
        "gstin_no": gstin_map.get(party_name, challan.get('gstin_no', '') or ""),
        "pan_no": challan.get('pan_no', '') or "",
    }
    challan['weaver_obj'] = {
        "name": weaver_name,
        "address": "",
        "gstin_no": gstin_map.get(weaver_name, ""),
        "pan_no": "",
    }
    challan['firm_obj'] = {
        "name": firm_name,
        "address": "",
        "gstin_no": gstin_map.get(firm_name, ""),
        "pan_no": "",
    }


def _clear_mill_identity_fields(challan: Dict[str, Any]) -> None:
    challan['party'] = ""
    challan['party_address'] = ""
    challan['firm'] = ""
    challan['gstin_no'] = ""
    challan['pan_no'] = ""
    challan['gstin_map'] = {}
    challan['weaver'] = ""


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
        "weaver_obj": _empty_entity_obj(),
        "party_obj": _empty_entity_obj(),
        "firm_obj": _empty_entity_obj(),
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

    supplier_section = decoded
    supplier_split = re.split(r'JOB\s+ISSUE\s+DELIVERY\s+CHALLAN', decoded, maxsplit=1, flags=re.IGNORECASE)
    if supplier_split:
        supplier_section = supplier_split[0]
    supplier_gstins = extract_all_gstins(supplier_section)

    firm_gstin = ""
    firm_match = re.search(
        r'TO[,\s]+JAI\s+MATA\s+DI\s+FASHIONS\s+PVT\.?\s*LTD.*?GSTIN[:\s]+([A-Z0-9]{13,16})',
        decoded,
        re.IGNORECASE | re.DOTALL,
    )
    if firm_match:
        firm_gstin = re.sub(r'[^A-Z0-9]', '', firm_match.group(1).upper())
    elif len(all_gstins) == 1:
        firm_gstin = all_gstins[0]

    supplier_gstin = ""
    if supplier_gstins:
        supplier_gstin = supplier_gstins[0]
    elif len(all_gstins) > 1:
        for gstin in all_gstins:
            if gstin != firm_gstin:
                supplier_gstin = gstin
                break

    c['gstin_no'] = supplier_gstin or ""

    # PAN No (format: 5 letters + 4 digits + 1 letter = 10 chars)
    pan_m = re.search(r'PAN\s*NO[:\s]+([A-Z]{5}[0-9]{4}[A-Z])', decoded, re.IGNORECASE)
    if pan_m:
        c['pan_no'] = pan_m.group(1).upper()

    if not supplier_gstin and c['pan_no']:
        supplier_gstin_m = re.search(
            r'GSTIN[:\s]*([0-9]{2})\S{10}([A-Z0-9]{3})',
            supplier_section,
            re.IGNORECASE,
        )
        if supplier_gstin_m:
            supplier_gstin = f"{supplier_gstin_m.group(1)}{c['pan_no']}{supplier_gstin_m.group(2)}"
            c['gstin_no'] = supplier_gstin

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
    if c['party'] and supplier_gstin:
        gstin_map[c['party']] = supplier_gstin
    if c['firm'] and firm_gstin:
        gstin_map[c['firm']] = firm_gstin

    remaining_gstins = [
        gstin for gstin in all_gstins
        if gstin and gstin not in {supplier_gstin, firm_gstin}
    ]
    if sub_party and remaining_gstins:
        gstin_map[sub_party] = remaining_gstins[0]
    c['gstin_map'] = gstin_map

    weaver_gstin = gstin_map.get(c['party'], "")
    firm_gstin = gstin_map.get(c['firm'], "")
    party_gstin = gstin_map.get(sub_party, "")
    _build_format_a_delivery_entities(
        c,
        weaver_name=c['party'],
        weaver_address=c['party_address'],
        weaver_gstin=weaver_gstin,
        weaver_pan=c['pan_no'],
        party_name=sub_party,
        party_gstin=party_gstin,
        firm_name=c['firm'],
        firm_gstin=firm_gstin,
    )
    _clear_delivery_identity_fields(c)

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
    header_present = bool(HEADER_RE.search(text))

    for line in text.split('\n'):
        stripped = line.strip()

        if HEADER_RE.search(stripped):
            in_table = True
            continue

        # Normalize comma-as-decimal (e.g. "120,00" -> "120.00")
        normalized = re.sub(r'(\d+),(\d{2})\b', r'\1.\2', stripped)
        nums = re.findall(r'\d+(?:\.\d+)?', normalized)

        # Auto-enter table mode if we see typical 3-column row data (Srl, Taka, Meter)
        # Only use this fallback when no explicit table header exists in the text.
        if not header_present and not in_table and len(nums) >= 3:
            try:
                sr = int(float(nums[0]))
                meter = float(nums[2])
                if 1 <= sr <= 500 and meter > 0:
                    in_table = True
            except ValueError:
                pass

        if not in_table or not nums:
            continue

        if re.search(r'Total\s+Taka|Remark|Received|Prepared|Authorised', stripped, re.IGNORECASE):
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
        if not header_present and len(nums) == 2:
            try:
                sr = int(float(nums[0]))
                meter = to_float(nums[1])
                if 1 <= sr <= 500 and meter > 0 and sr not in table_map:
                    table_map[sr] = (sr, meter)
            except (ValueError, IndexError):
                pass

        # Single number that looks like a meter reading (>5.0) -> assign to next sr
        elif not header_present and len(nums) == 1:
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


def _looks_like_delivery_party_value(value: str) -> bool:
    value = str(value or '').strip(' |:-')
    if not value or len(value) < 3:
        return False
    upper = value.upper()
    if re.search(r'DELIVERY|CHALLAN|DATE|QUALITY|BROC?KER|GSTIN|METER|TAKA|SRL?\.?\s*NO', upper):
        return False
    return bool(re.search(r'[A-Z]{3,}', upper))


def _extract_delivery_header_fields(raw_text: str, layout_text: str = '') -> Dict[str, str]:
    combined_text = layout_text or raw_text
    lines = [line.rstrip() for line in combined_text.split('\n') if line.strip()]
    result = {
        'party_name': '',
        'delivery_at': '',
        'quality': '',
        'date': '',
    }

    for i, line in enumerate(lines):
        if re.search(r'M/s\.?\s*Party.*Delivery\s+At', line, re.IGNORECASE):
            line_date_match = re.search(r'Date\s*[: ]\s*[/|\\-]*\s*([0-3]?\d[-/][01]?\d[-/](?:20)?\d{2,4}|\b20\d{2}\b)', line, re.IGNORECASE)
            if line_date_match:
                result['date'] = line_date_match.group(1).replace('-', '/')

            for candidate_line in lines[i + 1:i + 5]:
                if re.search(r'GSTIN|Srl?\.?\s*No|Taka\s*No|Total\s+Taka|Remark', candidate_line, re.IGNORECASE):
                    break

                columns = [col.strip(' |:-') for col in re.split(r'\s{2,}', candidate_line) if col.strip()]
                if not columns:
                    continue

                if not result['party_name'] and _looks_like_delivery_party_value(columns[0]):
                    result['party_name'] = columns[0]

                if not result['delivery_at']:
                    for col in columns[1:]:
                        if re.search(r'JAI\s*MATA|FASHIONS?\s*PVT', col, re.IGNORECASE):
                            result['delivery_at'] = col.strip()
                            break

                if not result['date']:
                    date_match = re.search(r'[/|\\-]*\s*([0-3]?\d[-/][01]?\d[-/](?:20)?\d{2,4}|\b20\d{2}\b)', candidate_line)
                    if date_match:
                        result['date'] = date_match.group(1).replace('-', '/')

                if not result['quality']:
                    quality_match = re.search(r'Quality\s*[:\s]+([A-Z0-9 ./_-]{3,40})', candidate_line, re.IGNORECASE)
                    if not quality_match:
                        quality_match = re.search(r'Qlty\s*[: ]?\s*([A-Z0-9 ./_-]{3,40})', candidate_line, re.IGNORECASE)
                    if quality_match:
                        result['quality'] = quality_match.group(1).strip(' |:-')
                    elif len(columns) >= 3 and _looks_like_delivery_party_value(columns[-1]):
                        result['quality'] = columns[-1].strip()
                    elif re.search(r'Quality', line, re.IGNORECASE) and len(columns) >= 3:
                        result['quality'] = columns[-1].strip()
            break

    if not result['party_name']:
        party_match = re.search(
            r'(?:M/s\.?\s*Party|M/s\.?)\s*[: ]\s*([A-Z0-9&.,/() \-]{3,80})',
            raw_text,
            re.IGNORECASE
        )
        if party_match:
            candidate = re.split(r'\s{2,}|DELIVERY\s+AT|CHALLAN\s+NO|DATE|QUALITY', party_match.group(1), flags=re.IGNORECASE)[0]
            candidate = candidate.strip(' |:-')
            if _looks_like_delivery_party_value(candidate):
                result['party_name'] = candidate

    if not result['delivery_at']:
        delivery_match = re.search(
            r'(JAI\s*MATA(?:DI)?\s*FASHIONS?\s*PVT\.?\s*LTD\.?(?:\s*\(UNIT-?\d+\))?)',
            combined_text,
            re.IGNORECASE
        )
        if delivery_match:
            result['delivery_at'] = delivery_match.group(1).strip(' |:-')

    return result


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

    # Prefer label-aware GST extraction for mill challans so party/firm GSTs do not get swapped.
    party_gstin = ""
    firm_gstin = ""

    party_gstin_m = re.search(r'GSTIN(?:/UIN)?\s*(?:NO\.?)?\s*[:\-]+\s*([A-Z0-9]{13,16})', text, re.IGNORECASE)
    if party_gstin_m:
        party_gstin = re.sub(r'[^A-Z0-9]', '', party_gstin_m.group(1).upper())

    firm_gstin_m = re.search(r'\bGST\s*:\s*([A-Z0-9]{13,16})', text, re.IGNORECASE)
    if firm_gstin_m:
        firm_gstin = re.sub(r'[^A-Z0-9]', '', firm_gstin_m.group(1).upper())

    all_gstins = extract_all_gstins(text)
    if not party_gstin and all_gstins:
        party_gstin = all_gstins[0]
    if not firm_gstin and len(all_gstins) > 1:
        firm_gstin = all_gstins[1]

    c['gstin_no'] = party_gstin or ""

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

    # Build gstin_map using label-aware GST extraction when available.
    gstin_map = {}
    if c['party'] and party_gstin:
        gstin_map[c['party']] = party_gstin
    if c['firm'] and firm_gstin:
        gstin_map[c['firm']] = firm_gstin
    c['gstin_map'] = gstin_map
    _build_mill_challan_entities(c)
    _clear_mill_identity_fields(c)

    c['table'] = parse_table_format_b(text)
    return c


def parse_format_c(raw_text: str, layout_text: str = '') -> Dict[str, Any]:
    """Parse Delivery Challan (DHARMIL TEXTILES / SHREEJI style — Srl.No./Taka No./Meter table)."""
    c = empty_challan()
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    header_fields = _extract_delivery_header_fields(raw_text, layout_text)

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
    c['weaver'] = c['party']

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
    if header_fields.get('quality') and _looks_like_noisy_quality(c.get('quality', '')):
        c['quality'] = header_fields['quality']

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

    c['ms_party'] = c['ms_party'] or header_fields.get('party_name', '')
    c['delivery_at'] = c['delivery_at'] or header_fields.get('delivery_at', '')
    if header_fields.get('date') and not c.get('date'):
        c['date'] = header_fields['date']
        c['challan_date'] = header_fields['date']

    # Build gstin_map after ms_party/delivery_at are set by layout parsing above
    gstin_map = {}
    firm_order = [c['party'], c['ms_party'], c['delivery_at']]
    for i, name in enumerate(firm_order):
        if name and i < len(all_gstins):
            gstin_map[name] = all_gstins[i]
    c['gstin_map'] = gstin_map
    _build_delivery_challan_entities(c)
    _clear_delivery_identity_fields(c)

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


def _parse_challan_from_text(fmt: str, text: str, layout_text: str = '') -> Dict[str, Any]:
    if fmt == 'A':
        return parse_format_a(text)
    if fmt == 'B':
        return parse_format_b(text)
    return parse_format_c(text, layout_text or text)


def _should_ocr_pdf_page(text: str, challan: Dict[str, Any]) -> bool:
    compact_text = re.sub(r'\s+', '', text or '')
    if len(compact_text) < 40:
        return True

    return not any([
        str(challan.get('challan_no', '')).strip(),
        str(challan.get('party', '')).strip(),
        challan.get('table'),
    ])


def _extract_challan_from_ocr_image(image: Image.Image) -> Dict[str, Any]:
    content_bytes, _mime_type = _image_to_bytes(_preprocess_image(image))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(content_bytes)
        tmp_path = tmp.name

    try:
        extracted = _extract_structured_challan_from_file(tmp_path)
        return _map_llama_extract_to_challan(extracted)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_challans_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    challans = []
    seen = set()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ''
            layout_text = page.extract_text(layout=True) or ''

            challan = {}
            if text.strip():
                fmt = detect_format(text)
                challan = _parse_challan_from_text(fmt, text, layout_text=layout_text)

            if not challan or _should_ocr_pdf_page(text, challan):
                page_image = page.to_image(resolution=200).original
                ocr_challan = _extract_challan_from_ocr_image(page_image)
                if challan:
                    challan = _merge_extracted_challans(challan, ocr_challan)
                else:
                    challan = ocr_challan

            if not challan:
                continue

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
        challan = _extract_challan_from_ocr_image(Image.open(file_path))
        return [challan] if challan else []
    else:
        raise ValueError(f"Unsupported file type: {file_ext}. Supported types: .pdf, .jpg, .jpeg, .png")
