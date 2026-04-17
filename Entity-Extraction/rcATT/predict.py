"""
Predict MITRE ATT&CK tactics and techniques from a CTI report.
Run from: cd /home/saqib/cti/STIXnet && python3 Entity-Extraction/rcATT/predict.py <path-to-report>
"""
import joblib
import pandas as pd
import re
import sys

from sklearn.base import BaseEstimator, TransformerMixin
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer

CODE_TACTICS = ['TA0043', 'TA0042', 'TA0001', 'TA0002', 'TA0003', 'TA0004', 'TA0005', 'TA0006', 'TA0007', 'TA0008', 'TA0009', 'TA0011', 'TA0010', 'TA0040']
NAME_TACTICS = ['Reconnaissance', 'Resource Development', 'Initial Access', 'Execution', 'Persistence', 'Privilege Escalation', 'Defense Evasion', 'Credential Access', 'Discovery', 'Lateral Movement', 'Collection', 'Command and Control', 'Exfiltration', 'Impact']

CODE_TECHNIQUES = ['T1595', 'T1592', 'T1589', 'T1590', 'T1591', 'T1598', 'T1597', 'T1596', 'T1593', 'T1594', 'T1583', 'T1586', 'T1584', 'T1587', 'T1585', 'T1588', 'T1608', 'T1189', 'T1190', 'T1133', 'T1200', 'T1566', 'T1091', 'T1195', 'T1199', 'T1078', 'T1059', 'T1609', 'T1610', 'T1203', 'T1559', 'T1106', 'T1053', 'T1129', 'T1072', 'T1569', 'T1204', 'T1047', 'T1098', 'T1197', 'T1547', 'T1037', 'T1176', 'T1554', 'T1136', 'T1543', 'T1546', 'T1574', 'T1525', 'T1556', 'T1137', 'T1542', 'T1505', 'T1205', 'T1548', 'T1134', 'T1484', 'T1611', 'T1068', 'T1055', 'T1612', 'T1622', 'T1140', 'T1006', 'T1480', 'T1211', 'T1222', 'T1564', 'T1562', 'T1070', 'T1202', 'T1036', 'T1578', 'T1112', 'T1601', 'T1599', 'T1027', 'T1647', 'T1620', 'T1207', 'T1014', 'T1553', 'T1218', 'T1216', 'T1221', 'T1127', 'T1535', 'T1550', 'T1497', 'T1600', 'T1220', 'T1557', 'T1110', 'T1555', 'T1212', 'T1187', 'T1606', 'T1056', 'T1111', 'T1621', 'T1040', 'T1003', 'T1528', 'T1558', 'T1539', 'T1552', 'T1087', 'T1010', 'T1217', 'T1580', 'T1538', 'T1526', 'T1619', 'T1613', 'T1482', 'T1083', 'T1615', 'T1046', 'T1135', 'T1201', 'T1120', 'T1069', 'T1057', 'T1012', 'T1018', 'T1518', 'T1082', 'T1614', 'T1016', 'T1049', 'T1033', 'T1007', 'T1124', 'T1210', 'T1534', 'T1570', 'T1563', 'T1021', 'T1080', 'T1560', 'T1123', 'T1119', 'T1185', 'T1115', 'T1530', 'T1602', 'T1213', 'T1005', 'T1039', 'T1025', 'T1074', 'T1114', 'T1113', 'T1125', 'T1071', 'T1092', 'T1132', 'T1001', 'T1568', 'T1573', 'T1008', 'T1105', 'T1104', 'T1095', 'T1571', 'T1572', 'T1090', 'T1219', 'T1102', 'T1020', 'T1030', 'T1048', 'T1041', 'T1011', 'T1052', 'T1567', 'T1029', 'T1537', 'T1531', 'T1485', 'T1486', 'T1565', 'T1491', 'T1561', 'T1499', 'T1495', 'T1490', 'T1498', 'T1496', 'T1489', 'T1529']
NAME_TECHNIQUES = ['Active Scanning', 'Gather Victim Host Information', 'Gather Victim Identity Information', 'Gather Victim Network Information', 'Gather Victim Org Information', 'Phishing for Information', 'Search Closed Sources', 'Search Open Technical Databases', 'Search Open Websites/Domains', 'Search Victim-Owned Websites', 'Acquire Infrastructure', 'Compromise Accounts', 'Compromise Infrastructure', 'Develop Capabilities', 'Establish Accounts', 'Obtain Capabilities', 'Stage Capabilities', 'Drive-by Compromise', 'Exploit Public-Facing Application', 'External Remote Services', 'Hardware Additions', 'Phishing', 'Replication Through Removable Media', 'Supply Chain Compromise', 'Trusted Relationship', 'Valid Accounts', 'Command and Scripting Interpreter', 'Container Administration Command', 'Deploy Container', 'Exploitation for Client Execution', 'Inter-Process Communication', 'Native API', 'Scheduled Task/Job', 'Shared Modules', 'Software Deployment Tools', 'System Services', 'User Execution', 'Windows Management Instrumentation', 'Account Manipulation', 'BITS Jobs', 'Boot or Logon Autostart Execution', 'Boot or Logon Initialization Scripts', 'Browser Extensions', 'Compromise Client Software Binary', 'Create Account', 'Create or Modify System Process', 'Event Triggered Execution', 'Hijack Execution Flow', 'Implant Internal Image', 'Modify Authentication Process', 'Office Application Startup', 'Pre-OS Boot', 'Server Software Component', 'Traffic Signaling', 'Abuse Elevation Control Mechanism', 'Access Token Manipulation', 'Domain Policy Modification', 'Escape to Host', 'Exploitation for Privilege Escalation', 'Process Injection', 'Build Image on Host', 'Debugger Evasion', 'Deobfuscate/Decode Files or Information', 'Direct Volume Access', 'Execution Guardrails', 'Exploitation for Defense Evasion', 'File and Directory Permissions Modification', 'Hide Artifacts', 'Impair Defenses', 'Indicator Removal on Host', 'Indirect Command Execution', 'Masquerading', 'Modify Cloud Compute Infrastructure', 'Modify Registry', 'Modify System Image', 'Network Boundary Bridging', 'Obfuscated Files or Information', 'Plist File Modification', 'Reflective Code Loading', 'Rogue Domain Controller', 'Rootkit', 'Subvert Trust Controls', 'Signed Binary Proxy Execution', 'Signed Script Proxy Execution', 'Template Injection', 'Trusted Developer Utilities Proxy Execution', 'Unused/Unsupported Cloud Regions', 'Use Alternate Authentication Material', 'Virtualization/Sandbox Evasion', 'Weaken Encryption', 'XSL Script Processing', 'Adversary-in-the-Middle', 'Brute Force', 'Credentials from Password Stores', 'Exploitation for Credential Access', 'Forced Authentication', 'Forge Web Credentials', 'Input Capture', 'Two-Factor Authentication Interception', 'Multi-Factor Authentication Request Generation', 'Network Sniffing', 'OS Credential Dumping', 'Steal Application Access Token', 'Steal or Forge Kerberos Tickets', 'Steal Web Session Cookie', 'Unsecured Credentials', 'Account Discovery', 'Application Window Discovery', 'Browser Bookmark Discovery', 'Cloud Infrastructure Discovery', 'Cloud Service Dashboard', 'Cloud Service Discovery', 'Cloud Storage Object Discovery', 'Container and Resource Discovery', 'Domain Trust Discovery', 'File and Directory Discovery', 'Group Policy Discovery', 'Network Service Scanning', 'Network Share Discovery', 'Password Policy Discovery', 'Peripheral Device Discovery', 'Permission Groups Discovery', 'Process Discovery', 'Query Registry', 'Remote System Discovery', 'Software Discovery', 'System Information Discovery', 'System Location Discovery', 'System Network Configuration Discovery', 'System Network Connections Discovery', 'System Owner/User Discovery', 'System Service Discovery', 'System Time Discovery', 'Exploitation of Remote Services', 'Internal Spearphishing', 'Lateral Tool Transfer', 'Remote Service Session Hijacking', 'Remote Services', 'Taint Shared Content', 'Archive Collected Data', 'Audio Capture', 'Automated Collection', 'Browser Session Hijacking', 'Clipboard Data', 'Data from Cloud Storage Object', 'Data from Configuration Repository', 'Data from Information Repositories', 'Data from Local System', 'Data from Network Shared Drive', 'Data from Removable Media', 'Data Staged', 'Email Collection', 'Screen Capture', 'Video Capture', 'Application Layer Protocol', 'Communication Through Removable Media', 'Data Encoding', 'Data Obfuscation', 'Dynamic Resolution', 'Encrypted Channel', 'Fallback Channels', 'Ingress Tool Transfer', 'Multi-Stage Channels', 'Non-Application Layer Protocol', 'Non-Standard Port', 'Protocol Tunneling', 'Proxy', 'Remote Access Software', 'Web Service', 'Automated Exfiltration', 'Data Transfer Size Limits', 'Exfiltration Over Alternative Protocol', 'Exfiltration Over C2 Channel', 'Exfiltration Over Other Network Medium', 'Exfiltration Over Physical Medium', 'Exfiltration Over Web Service', 'Scheduled Transfer', 'Transfer Data to Cloud Account', 'Account Access Removal', 'Data Destruction', 'Data Encrypted for Impact', 'Data Manipulation', 'Defacement', 'Disk Wipe', 'Endpoint Denial of Service', 'Firmware Corruption', 'Inhibit System Recovery', 'Network Denial of Service', 'Resource Hijacking', 'Service Stop', 'System Shutdown/Reboot']

TEXT_FEATURES = ['processed']


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\r\n", "\t", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def processing(df):
    df['processed'] = df['Text'].map(lambda com: clean_text(com))
    return df


class StemTokenizer(object):
    def __init__(self):
        self.st = EnglishStemmer()
    def __call__(self, doc):
        return [self.st.stem(t) for t in word_tokenize(doc)]


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 Entity-Extraction/rcATT/predict.py <path-to-cti-report>")
        print("Example: python3 Entity-Extraction/rcATT/predict.py Dataset/Data/APT28.txt")
        sys.exit(1)

    filepath = sys.argv[1]
    with open(filepath, 'r') as f:
        text = f.read()

    # Load models
    pipeline_tactics = joblib.load('./Entity-Extraction/rcATT/Models/tactics.joblib')
    pipeline_techniques = joblib.load('./Entity-Extraction/rcATT/Models/techniques.joblib')

    # Prepare input
    report = processing(pd.DataFrame([text], columns=['Text']))[TEXT_FEATURES]

    # Predict
    pred_tactics = pipeline_tactics.predict(report)
    predprob_tactics = pipeline_tactics.decision_function(report)
    pred_techniques = pipeline_techniques.predict(report)
    predprob_techniques = pipeline_techniques.decision_function(report)

    import numpy as np

    print(f"\n=== Predicted ATT&CK TTPs from: {filepath} ===")

    # Show strict predictions (score > 0) first, then top-N if none found
    print("\n  TACTICS:")
    found_tactics = False
    for i, val in enumerate(pred_tactics[0]):
        if val == 1:
            print(f"    {CODE_TACTICS[i]} - {NAME_TACTICS[i]} (confidence: {predprob_tactics[0][i]:.3f})")
            found_tactics = True
    if not found_tactics:
        print("    (none above threshold, showing top 5 by score)")
        top_idx = np.argsort(predprob_tactics[0])[::-1][:5]
        for i in top_idx:
            print(f"    {CODE_TACTICS[i]} - {NAME_TACTICS[i]} (score: {predprob_tactics[0][i]:.3f})")

    print("\n  TECHNIQUES:")
    found_techniques = False
    for i, val in enumerate(pred_techniques[0]):
        if val == 1:
            print(f"    {CODE_TECHNIQUES[i]} - {NAME_TECHNIQUES[i]} (confidence: {predprob_techniques[0][i]:.3f})")
            found_techniques = True
    if not found_techniques:
        print("    (none above threshold, showing top 10 by score)")
        top_idx = np.argsort(predprob_techniques[0])[::-1][:10]
        for i in top_idx:
            print(f"    {CODE_TECHNIQUES[i]} - {NAME_TECHNIQUES[i]} (score: {predprob_techniques[0][i]:.3f})")
