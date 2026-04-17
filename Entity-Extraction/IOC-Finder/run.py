import json
import sys
from ioc_finder import find_iocs

if len(sys.argv) < 2:
    print("Usage: python3 run.py <path-to-cti-report>")
    print("Example: python3 run.py /home/saqib/cti/STIXnet/Dataset/Data/APT28.txt")
    sys.exit(1)

filepath = sys.argv[1]

with open(filepath, 'r') as f:
    text = f.read()

iocs, pos_map = find_iocs(text)

print(f"\n=== Extracted IOCs from: {filepath} ===\n")

found = False
for ioc_type, values in iocs.items():
    if isinstance(values, dict):
        for sub_type, sub_values in values.items():
            if sub_values:
                print(f"  {ioc_type} ({sub_type}): {sub_values}")
                found = True
    elif values:
        print(f"  {ioc_type}: {values}")
        found = True

if not found:
    print("  No IOCs found.")
