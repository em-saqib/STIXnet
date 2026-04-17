import csv
import re
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python3 run.py <path-to-cti-report>")
    print("Example: python3 run.py ../../Dataset/Data/APT28.txt")
    sys.exit(1)

# Load nationalities lookup
csv_path = os.path.join(os.path.dirname(__file__), 'nationalities.csv')
lookup = {}
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        lookup[row['Nationality']] = row['Nation']

# Read the CTI report
filepath = sys.argv[1]
with open(filepath, 'r') as f:
    text = f.read()

# Search for each nationality in the text (case-sensitive, whole word match)
matches = {}
for nationality, nation in lookup.items():
    pattern = r'\b' + re.escape(nationality) + r'\b'
    found = re.findall(pattern, text)
    if found:
        if nation not in matches:
            matches[nation] = []
        matches[nation].append((nationality, len(found)))

print(f"\n=== Extracted Locations from: {filepath} ===\n")

if matches:
    for nation, hits in sorted(matches.items()):
        for nationality, count in hits:
            print(f"  {nationality} -> {nation} (found {count}x)")
else:
    print("  No locations found.")
