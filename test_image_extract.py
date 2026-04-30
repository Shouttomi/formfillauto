from extractor import extract_challans


def assert_image_fields(path: str, required_fields):
    challans = extract_challans(path)
    assert challans, f"No challans extracted from {path}"
    challan = challans[0]
    for field in required_fields:
        value = challan.get(field)
        assert value not in ("", 0, 0.0, None, [], {}), f"Missing `{field}` for {path}: {challan}"


assert_image_fields(
    "pdf/SHREEJITEX54.jpeg",
    ["party", "date", "quality", "meter", "ms_party", "remark"],
)

assert_image_fields(
    "pdf/VRAJSHOP.jpeg",
    ["party", "challan_no", "quality", "meter", "gstin_no"],
)
