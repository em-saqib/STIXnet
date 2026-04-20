#!/bin/bash
# STIXnet Pipeline - Run all extraction steps on a CTI report
# Usage: ./run_pipeline.sh <path-to-cti-report>
# Example: ./run_pipeline.sh Dataset/Data/APT28.txt

if [ -z "$1" ]; then
    echo "Usage: ./run_pipeline.sh <path-to-cti-report>"
    echo "Example: ./run_pipeline.sh Dataset/Data/APT28.txt"
    exit 1
fi

REPORT="$1"
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASEDIR"

if [ ! -f "$REPORT" ]; then
    echo "Error: File '$REPORT' not found."
    exit 1
fi

echo "============================================================"
echo "  STIXnet Pipeline"
echo "  Report: $REPORT"
echo "============================================================"

echo ""
echo ">>> STEP 1: IOC Finder (IPs, hashes, CVEs, domains, etc.)"
echo "------------------------------------------------------------"
cd "$BASEDIR/Entity-Extraction/IOC-Finder"
python3 run.py "$BASEDIR/$REPORT"

echo ""
echo ">>> STEP 2: Knowledge Base (Locations)"
echo "------------------------------------------------------------"
cd "$BASEDIR/Entity-Extraction/Knowledge-Base"
python3 run.py "$BASEDIR/$REPORT"

echo ""
echo ">>> STEP 3: rcATT (MITRE ATT&CK Tactics & Techniques)"
echo "------------------------------------------------------------"
cd "$BASEDIR"
python3 Entity-Extraction/rcATT/predict.py "$REPORT" 2>&1 | grep -v Warning | grep -v warn

echo ""
echo ">>> STEP 4a: Regenerate STIX relationship schema (Relations.csv)"
echo "------------------------------------------------------------"
cd "$BASEDIR"
python3 Relation-Extraction/relationsConversion.py 2>&1 | grep -v nltk_data
echo "  Relations.csv generated ($(wc -l < Relation-Extraction/Relations.csv) rules)"

echo ""
echo ">>> STEP 4b: Extract per-report STIX relations"
echo "------------------------------------------------------------"
python3 Relation-Extraction/extract_relations.py "$REPORT" 2>&1 | grep -v nltk_data

echo ""
echo "============================================================"
echo "  Pipeline complete."
echo "============================================================"
