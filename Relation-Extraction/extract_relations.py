"""
Extract STIX relationships from a CTI report.

For each sentence: find entities (IOCs, locations, ATT&CK techniques),
map them to STIX types, and emit every (src, type, dst) triple whose
type-pair is allowed by Relations.csv (derived from SROs.csv, the
official STIX relationship schema).

Output: Output/<input-name>.csv  with columns:
    src_type,src_value,relationship,dst_type,dst_value,sentence
"""
import csv
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'Entity-Extraction', 'IOC-Finder'))

from ioc_finder import find_iocs  # noqa: E402


# --- Mapping IOC Finder keys -> STIX types -----------------------------------
IOC_TYPE_MAP = {
    'ipv4s': 'ipv4-addr',
    'ipv6s': 'ipv6-addr',
    'domains': 'domain-name',
    'urls': 'url',
    'email_addresses': 'email-addr',
    'email_addresses_complete': 'email-addr',
    'md5s': 'file',
    'sha1s': 'file',
    'sha256s': 'file',
    'sha512s': 'file',
    'imphashes': 'file',
    'authentihashes': 'file',
    'ssdeeps': 'file',
    'file_paths': 'file',
    'cves': 'vulnerability',
    'attack_techniques': 'attack-pattern',
    'attack_mitigations': 'course-of-action',
    'registry_key_paths': 'windows-registry-key',
    'mac_addresses': 'mac-addr',
    'asns': 'autonomous-system',
}


def load_relation_schema():
    """Build {(src_type, dst_type): [relationship_verbs]} from Relations.csv."""
    rel_path = os.path.join(SCRIPT_DIR, 'Relations.csv')
    schema = {}
    with open(rel_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['src'], row['dst'])
            schema.setdefault(key, []).append(row['type'])
    return schema


def load_locations():
    """nationality -> nation from Knowledge-Base/nationalities.csv."""
    path = os.path.join(REPO_ROOT, 'Entity-Extraction', 'Knowledge-Base', 'nationalities.csv')
    lookup = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row['Nationality']] = row['Nation']
    return lookup


def split_sentences(text):
    """Return list of (sentence, start_offset, end_offset)."""
    pattern = re.compile(r'[^.!?\n]+[.!?\n]?', re.MULTILINE)
    spans = []
    for m in pattern.finditer(text):
        s = m.group().strip()
        if s:
            spans.append((s, m.start(), m.end()))
    return spans


def collect_entities(text, nationality_lookup, report_level_techniques=None):
    """
    Return list of entity dicts:
      {'stix_type', 'value', 'start', 'end', 'scope'}
    scope='sentence' means sentence-level match; scope='report' means
    the entity was predicted for the whole report and should pair with
    entities in every sentence.
    """
    entities = []

    iocs, pos_map = find_iocs(text)
    for ioc_key, stix_type in IOC_TYPE_MAP.items():
        positions = pos_map.get(ioc_key, {}) if pos_map else {}
        for value, ranges in positions.items():
            for start, end in ranges:
                entities.append({
                    'stix_type': stix_type,
                    'value': value,
                    'start': start,
                    'end': end,
                    'scope': 'sentence',
                })

    for nationality, nation in nationality_lookup.items():
        for m in re.finditer(r'\b' + re.escape(nationality) + r'\b', text):
            entities.append({
                'stix_type': 'location',
                'value': nation,
                'start': m.start(),
                'end': m.end(),
                'scope': 'sentence',
            })

    if report_level_techniques:
        for code, name in report_level_techniques:
            entities.append({
                'stix_type': 'attack-pattern',
                'value': f'{code} ({name})',
                'start': -1,
                'end': -1,
                'scope': 'report',
            })

    return entities


def group_by_sentence(entities, sentences):
    """Return list of (sentence_text, [entities_in_sentence]).

    Report-level entities are added to every sentence so they can pair
    with sentence-local entities.
    """
    report_level = [e for e in entities if e.get('scope') == 'report']
    sentence_level = [e for e in entities if e.get('scope') != 'report']
    grouped = []
    for sent, s_start, s_end in sentences:
        in_sent = [e for e in sentence_level if s_start <= e['start'] < s_end]
        if not in_sent:
            continue
        combined = in_sent + report_level
        if len(combined) >= 2:
            grouped.append((sent, combined))
    return grouped


def extract_relations(sentences_with_entities, schema):
    """Yield (src_type, src_value, rel, dst_type, dst_value, sentence)."""
    rows = []
    seen = set()
    for sent, ents in sentences_with_entities:
        for i, a in enumerate(ents):
            for j, b in enumerate(ents):
                if i == j:
                    continue
                key_pair = (a['stix_type'], b['stix_type'])
                for rel in schema.get(key_pair, []):
                    dedup = (a['stix_type'], a['value'], rel, b['stix_type'], b['value'], sent)
                    if dedup in seen:
                        continue
                    seen.add(dedup)
                    rows.append(dedup)
    return rows


def predict_techniques(text):
    """Run rcATT to get report-level attack-pattern predictions.
    Returns list of (code, name) tuples. Best-effort; returns [] on failure.
    """
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, 'Entity-Extraction', 'rcATT'))
        import predict as rcatt  # noqa
        import joblib
        import pandas as pd
        import numpy as np

        # joblib pickles reference these classes by their original module (__main__
        # when predict.py is run directly). Alias them here so unpickling works.
        main_mod = sys.modules['__main__']
        for cls in ('TextSelector', 'StemTokenizer', 'LemmaTokenizer'):
            setattr(main_mod, cls, getattr(rcatt, cls))
        main_mod.clean_text = rcatt.clean_text
        main_mod.processing = rcatt.processing

        pipeline = joblib.load(os.path.join(REPO_ROOT, 'Entity-Extraction', 'rcATT', 'Models', 'techniques.joblib'))
        df = rcatt.processing(pd.DataFrame([text], columns=['Text']))[rcatt.TEXT_FEATURES]
        pred = pipeline.predict(df)
        scores = pipeline.decision_function(df)
        results = []
        for i, v in enumerate(pred[0]):
            if v == 1:
                results.append((rcatt.CODE_TECHNIQUES[i], rcatt.NAME_TECHNIQUES[i]))
        if not results:
            top = np.argsort(scores[0])[::-1][:5]
            for i in top:
                results.append((rcatt.CODE_TECHNIQUES[i], rcatt.NAME_TECHNIQUES[i]))
        return results
    except Exception as e:
        print(f"  (rcATT prediction skipped: {e})", file=sys.stderr)
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 Relation-Extraction/extract_relations.py <path-to-cti-report>")
        sys.exit(1)

    report_path = sys.argv[1]
    with open(report_path) as f:
        text = f.read()

    schema = load_relation_schema()
    nat_lookup = load_locations()
    techniques = predict_techniques(text)

    sentences = split_sentences(text)
    entities = collect_entities(text, nat_lookup, report_level_techniques=techniques)
    grouped = group_by_sentence(entities, sentences)
    rows = extract_relations(grouped, schema)

    out_dir = os.path.join(REPO_ROOT, 'Output')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(report_path))[0]
    out_path = os.path.join(out_dir, base + '.csv')

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['src_type', 'src_value', 'relationship', 'dst_type', 'dst_value', 'sentence'])
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} relations to {out_path}")


if __name__ == '__main__':
    main()
