from extractor import extract_challans
import json

result = extract_challans('pdf/VRAJSHOP.jpeg')
c = result[0]
with open('final_output2.txt', 'w', encoding='utf-8') as f:
    f.write(f"party      : {c['party']}\n")
    f.write(f"challan_no : {c['challan_no']}\n")
    f.write(f"date       : {c['date']}\n")
    f.write(f"quality    : {c['quality']}\n")
    f.write(f"taka       : {c['taka']}\n")
    f.write(f"meter      : {c['meter']}\n")
    f.write(f"gstin_no   : {c['gstin_no']}\n")
    f.write(f"table rows : {len(c['table'])}\n")
    for row in c['table']:
        f.write(f"  {row}\n")
