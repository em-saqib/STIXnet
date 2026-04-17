"""
Training script for rcATT - extracted from training.ipynb
Trains SVM classifiers for MITRE ATT&CK tactics and techniques prediction.
Run from: cd /home/saqib/cti/STIXnet && python3 Entity-Extraction/rcATT/train.py
"""
import joblib
import pandas as pd
import re
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

TEXT_FEATURES = ['processed']

CODE_TACTICS = ['TA0043', 'TA0042', 'TA0001', 'TA0002', 'TA0003', 'TA0004', 'TA0005', 'TA0006', 'TA0007', 'TA0008', 'TA0009', 'TA0011', 'TA0010', 'TA0040']
NAME_TACTICS = ['Reconnaissance', 'Resource Development', 'Initial Access', 'Execution', 'Persistence', 'Privilege Escalation', 'Defense Evasion', 'Credential Access', 'Discovery', 'Lateral Movement', 'Collection', 'Command and Control', 'Exfiltration', 'Impact']

CODE_TECHNIQUES = ['T1595', 'T1592', 'T1589', 'T1590', 'T1591', 'T1598', 'T1597', 'T1596', 'T1593', 'T1594', 'T1583', 'T1586', 'T1584', 'T1587', 'T1585', 'T1588', 'T1608', 'T1189', 'T1190', 'T1133', 'T1200', 'T1566', 'T1091', 'T1195', 'T1199', 'T1078', 'T1059', 'T1609', 'T1610', 'T1203', 'T1559', 'T1106', 'T1053', 'T1129', 'T1072', 'T1569', 'T1204', 'T1047', 'T1098', 'T1197', 'T1547', 'T1037', 'T1176', 'T1554', 'T1136', 'T1543', 'T1546', 'T1574', 'T1525', 'T1556', 'T1137', 'T1542', 'T1505', 'T1205', 'T1548', 'T1134', 'T1484', 'T1611', 'T1068', 'T1055', 'T1612', 'T1622', 'T1140', 'T1006', 'T1480', 'T1211', 'T1222', 'T1564', 'T1562', 'T1070', 'T1202', 'T1036', 'T1578', 'T1112', 'T1601', 'T1599', 'T1027', 'T1647', 'T1620', 'T1207', 'T1014', 'T1553', 'T1218', 'T1216', 'T1221', 'T1127', 'T1535', 'T1550', 'T1497', 'T1600', 'T1220', 'T1557', 'T1110', 'T1555', 'T1212', 'T1187', 'T1606', 'T1056', 'T1111', 'T1621', 'T1040', 'T1003', 'T1528', 'T1558', 'T1539', 'T1552', 'T1087', 'T1010', 'T1217', 'T1580', 'T1538', 'T1526', 'T1619', 'T1613', 'T1482', 'T1083', 'T1615', 'T1046', 'T1135', 'T1201', 'T1120', 'T1069', 'T1057', 'T1012', 'T1018', 'T1518', 'T1082', 'T1614', 'T1016', 'T1049', 'T1033', 'T1007', 'T1124', 'T1210', 'T1534', 'T1570', 'T1563', 'T1021', 'T1080', 'T1560', 'T1123', 'T1119', 'T1185', 'T1115', 'T1530', 'T1602', 'T1213', 'T1005', 'T1039', 'T1025', 'T1074', 'T1114', 'T1113', 'T1125', 'T1071', 'T1092', 'T1132', 'T1001', 'T1568', 'T1573', 'T1008', 'T1105', 'T1104', 'T1095', 'T1571', 'T1572', 'T1090', 'T1219', 'T1102', 'T1020', 'T1030', 'T1048', 'T1041', 'T1011', 'T1052', 'T1567', 'T1029', 'T1537', 'T1531', 'T1485', 'T1486', 'T1565', 'T1491', 'T1561', 'T1499', 'T1495', 'T1490', 'T1498', 'T1496', 'T1489', 'T1529']
NAME_TECHNIQUES = ['Active Scanning', 'Gather Victim Host Information', 'Gather Victim Identity Information', 'Gather Victim Network Information', 'Gather Victim Org Information', 'Phishing for Information', 'Search Closed Sources', 'Search Open Technical Databases', 'Search Open Websites/Domains', 'Search Victim-Owned Websites', 'Acquire Infrastructure', 'Compromise Accounts', 'Compromise Infrastructure', 'Develop Capabilities', 'Establish Accounts', 'Obtain Capabilities', 'Stage Capabilities', 'Drive-by Compromise', 'Exploit Public-Facing Application', 'External Remote Services', 'Hardware Additions', 'Phishing', 'Replication Through Removable Media', 'Supply Chain Compromise', 'Trusted Relationship', 'Valid Accounts', 'Command and Scripting Interpreter', 'Container Administration Command', 'Deploy Container', 'Exploitation for Client Execution', 'Inter-Process Communication', 'Native API', 'Scheduled Task/Job', 'Shared Modules', 'Software Deployment Tools', 'System Services', 'User Execution', 'Windows Management Instrumentation', 'Account Manipulation', 'BITS Jobs', 'Boot or Logon Autostart Execution', 'Boot or Logon Initialization Scripts', 'Browser Extensions', 'Compromise Client Software Binary', 'Create Account', 'Create or Modify System Process', 'Event Triggered Execution', 'Hijack Execution Flow', 'Implant Internal Image', 'Modify Authentication Process', 'Office Application Startup', 'Pre-OS Boot', 'Server Software Component', 'Traffic Signaling', 'Abuse Elevation Control Mechanism', 'Access Token Manipulation', 'Domain Policy Modification', 'Escape to Host', 'Exploitation for Privilege Escalation', 'Process Injection', 'Build Image on Host', 'Debugger Evasion', 'Deobfuscate/Decode Files or Information', 'Direct Volume Access', 'Execution Guardrails', 'Exploitation for Defense Evasion', 'File and Directory Permissions Modification', 'Hide Artifacts', 'Impair Defenses', 'Indicator Removal on Host', 'Indirect Command Execution', 'Masquerading', 'Modify Cloud Compute Infrastructure', 'Modify Registry', 'Modify System Image', 'Network Boundary Bridging', 'Obfuscated Files or Information', 'Plist File Modification', 'Reflective Code Loading', 'Rogue Domain Controller', 'Rootkit', 'Subvert Trust Controls', 'Signed Binary Proxy Execution', 'Signed Script Proxy Execution', 'Template Injection', 'Trusted Developer Utilities Proxy Execution', 'Unused/Unsupported Cloud Regions', 'Use Alternate Authentication Material', 'Virtualization/Sandbox Evasion', 'Weaken Encryption', 'XSL Script Processing', 'Adversary-in-the-Middle', 'Brute Force', 'Credentials from Password Stores', 'Exploitation for Credential Access', 'Forced Authentication', 'Forge Web Credentials', 'Input Capture', 'Two-Factor Authentication Interception', 'Multi-Factor Authentication Request Generation', 'Network Sniffing', 'OS Credential Dumping', 'Steal Application Access Token', 'Steal or Forge Kerberos Tickets', 'Steal Web Session Cookie', 'Unsecured Credentials', 'Account Discovery', 'Application Window Discovery', 'Browser Bookmark Discovery', 'Cloud Infrastructure Discovery', 'Cloud Service Dashboard', 'Cloud Service Discovery', 'Cloud Storage Object Discovery', 'Container and Resource Discovery', 'Domain Trust Discovery', 'File and Directory Discovery', 'Group Policy Discovery', 'Network Service Scanning', 'Network Share Discovery', 'Password Policy Discovery', 'Peripheral Device Discovery', 'Permission Groups Discovery', 'Process Discovery', 'Query Registry', 'Remote System Discovery', 'Software Discovery', 'System Information Discovery', 'System Location Discovery', 'System Network Configuration Discovery', 'System Network Connections Discovery', 'System Owner/User Discovery', 'System Service Discovery', 'System Time Discovery', 'Exploitation of Remote Services', 'Internal Spearphishing', 'Lateral Tool Transfer', 'Remote Service Session Hijacking', 'Remote Services', 'Taint Shared Content', 'Archive Collected Data', 'Audio Capture', 'Automated Collection', 'Browser Session Hijacking', 'Clipboard Data', 'Data from Cloud Storage Object', 'Data from Configuration Repository', 'Data from Information Repositories', 'Data from Local System', 'Data from Network Shared Drive', 'Data from Removable Media', 'Data Staged', 'Email Collection', 'Screen Capture', 'Video Capture', 'Application Layer Protocol', 'Communication Through Removable Media', 'Data Encoding', 'Data Obfuscation', 'Dynamic Resolution', 'Encrypted Channel', 'Fallback Channels', 'Ingress Tool Transfer', 'Multi-Stage Channels', 'Non-Application Layer Protocol', 'Non-Standard Port', 'Protocol Tunneling', 'Proxy', 'Remote Access Software', 'Web Service', 'Automated Exfiltration', 'Data Transfer Size Limits', 'Exfiltration Over Alternative Protocol', 'Exfiltration Over C2 Channel', 'Exfiltration Over Other Network Medium', 'Exfiltration Over Physical Medium', 'Exfiltration Over Web Service', 'Scheduled Transfer', 'Transfer Data to Cloud Account', 'Account Access Removal', 'Data Destruction', 'Data Encrypted for Impact', 'Data Manipulation', 'Defacement', 'Disk Wipe', 'Endpoint Denial of Service', 'Firmware Corruption', 'Inhibit System Recovery', 'Network Denial of Service', 'Resource Hijacking', 'Service Stop', 'System Shutdown/Reboot']

ALL_TTPS = CODE_TACTICS + CODE_TECHNIQUES

TACTICS_TECHNIQUES_RELATIONSHIP_DF = pd.DataFrame({'TA0043': pd.Series(['T1595', 'T1592', 'T1589', 'T1590', 'T1591', 'T1598', 'T1597', 'T1596', 'T1593', 'T1594']),
                                        'TA0042': pd.Series(['T1583', 'T1586', 'T1584', 'T1587', 'T1585', 'T1588', 'T1608']),
                                        'TA0001': pd.Series(['T1189', 'T1190', 'T1133', 'T1200', 'T1566', 'T1091', 'T1195', 'T1199', 'T1078']),
                                        'TA0002': pd.Series(['T1059', 'T1609', 'T1610', 'T1203', 'T1559', 'T1106', 'T1053', 'T1129', 'T1072', 'T1569', 'T1204', 'T1047']),
                                        'TA0003': pd.Series(['T1098', 'T1197', 'T1547', 'T1037', 'T1176', 'T1554', 'T1136', 'T1543', 'T1546', 'T1133', 'T1574', 'T1525', 'T1556', 'T1137', 'T1542', 'T1053', 'T1505', 'T1205', 'T1078']),
                                        'TA0004': pd.Series(['T1548', 'T1134', 'T1547', 'T1037', 'T1543', 'T1484', 'T1611', 'T1546', 'T1068', 'T1574', 'T1055', 'T1053', 'T1078']),
                                        'TA0005': pd.Series(['T1548', 'T1134', 'T1197', 'T1612', 'T1622', 'T1140', 'T1610', 'T1006', 'T1484', 'T1480', 'T1211', 'T1222', 'T1564', 'T1574', 'T1562', 'T1070', 'T1202', 'T1036', 'T1556', 'T1578', 'T1112', 'T1601', 'T1599', 'T1027', 'T1647', 'T1542', 'T1055', 'T1620', 'T1207', 'T1014', 'T1553', 'T1218', 'T1216', 'T1221', 'T1205', 'T1127', 'T1535', 'T1550', 'T1078', 'T1497', 'T1600', 'T1220']),
                                        'TA0006': pd.Series(['T1557', 'T1110', 'T1555', 'T1212', 'T1187', 'T1606', 'T1056', 'T1556', 'T1111', 'T1621', 'T1040', 'T1003', 'T1528', 'T1558', 'T1539', 'T1552']),
                                        'TA0007': pd.Series(['T1087', 'T1010', 'T1217', 'T1580', 'T1538', 'T1526', 'T1619', 'T1613', 'T1622', 'T1482', 'T1083', 'T1615', 'T1046', 'T1135', 'T1040', 'T1201', 'T1120', 'T1069', 'T1057', 'T1012', 'T1018', 'T1518', 'T1082', 'T1614', 'T1016', 'T1049', 'T1033', 'T1007', 'T1124', 'T1497']),
                                        'TA0008': pd.Series(['T1210', 'T1534', 'T1570', 'T1563', 'T1021', 'T1091', 'T1072', 'T1080', 'T1550']),
                                        'TA0009': pd.Series(['T1557', 'T1560', 'T1123', 'T1119', 'T1185', 'T1115', 'T1530', 'T1602', 'T1213', 'T1005', 'T1039', 'T1025', 'T1074', 'T1114', 'T1056', 'T1113', 'T1125']),
                                        'TA0011': pd.Series(['T1071', 'T1092', 'T1132', 'T1001', 'T1568', 'T1573', 'T1008', 'T1105', 'T1104', 'T1095', 'T1571', 'T1572', 'T1090', 'T1219', 'T1205', 'T1102']),
                                        'TA0010': pd.Series(['T1020', 'T1030', 'T1048', 'T1041', 'T1011', 'T1052', 'T1567', 'T1029', 'T1537']),
                                        'TA0040': pd.Series(['T1531', 'T1485', 'T1486', 'T1565', 'T1491', 'T1561', 'T1499', 'T1495', 'T1490', 'T1498', 'T1496', 'T1489', 'T1529'])
                                        })


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub("\r\n", "\t", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.)\{3\}(?:25[0-5] |2[0-4][0-9]|[01]?[0-9][0-9]?)(/([0-2][0-9]|3[0-2]|[0-9]))?', 'IPv4', text)
    text = re.sub('\b(CVE\-[0-9]{4}\-[0-9]{4,6})\b', 'CVE', text)
    text = re.sub('\b([a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)\b', 'email', text)
    text = re.sub('\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', 'IP', text)
    text = re.sub('\b([a-f0-9]{32}|[A-F0-9]{32})\b', 'MD5', text)
    text = re.sub('\b((HKLM|HKCU)\\[\\A-Za-z0-9-_]+)\b', 'registry', text)
    text = re.sub('\b([a-f0-9]{40}|[A-F0-9]{40})\b', 'SHA1', text)
    text = re.sub('\b([a-f0-9]{64}|[A-F0-9]{64})\b', 'SHA250', text)
    text = re.sub('http(s)?:\\[0-9a-zA-Z_\.\-\\]+.', 'URL', text)
    text = re.sub('CVE-[0-9]{4}-[0-9]{4,6}', 'vulnerability', text)
    text = re.sub('[a-zA-Z]{1}:\\[0-9a-zA-Z_\.\-\\]+', 'file', text)
    text = re.sub('\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b', 'hash', text)
    text = re.sub('x[A-Fa-f0-9]{2}', ' ', text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


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


def confidence_propagation_single(tactics_confidence_list, technique_name, technique_confidence_score):
    new_confidence_score = technique_confidence_score
    for tactic in CODE_TACTICS:
        if not TACTICS_TECHNIQUES_RELATIONSHIP_DF.loc[TACTICS_TECHNIQUES_RELATIONSHIP_DF[tactic] == technique_name].empty:
            lambdaim = 1/(np.exp(abs(technique_confidence_score-tactics_confidence_list[tactic])))
            new_confidence_score = new_confidence_score + lambdaim * tactics_confidence_list[tactic]
    return new_confidence_score


def confidence_propagation(predprob_tactics, pred_techniques, predprob_techniques):
    pred_techniques_corrected = pred_techniques
    predprob_techniques_corrected = predprob_techniques
    tactics_confidence_df = pd.DataFrame(data=predprob_tactics, columns=CODE_TACTICS)
    for j in range(len(predprob_techniques[0])):
        for i in range(len(predprob_techniques)):
            predprob_techniques_corrected[i][j] = confidence_propagation_single(tactics_confidence_df[i:(i+1)], CODE_TECHNIQUES[j], predprob_techniques[i][j])
            if predprob_techniques_corrected[i][j] >= float(0):
                pred_techniques_corrected[i][j] = int(1)
            else:
                pred_techniques_corrected[i][j] = int(0)
    return pred_techniques_corrected, predprob_techniques_corrected


def hanging_node(pred_tactics, predprob_tactics, pred_techniques, predprob_techniques, c, d):
    predprob_techniques_corrected = pred_techniques
    for i in range(len(pred_techniques)):
        for j in range(len(pred_techniques[0])):
            for k in range(len(pred_tactics[0])):
                if not TACTICS_TECHNIQUES_RELATIONSHIP_DF.loc[TACTICS_TECHNIQUES_RELATIONSHIP_DF[CODE_TACTICS[k]] == CODE_TECHNIQUES[j]].empty:
                    if predprob_techniques[i][j] < c and predprob_techniques[i][j] > 0 and predprob_tactics[i][k] < d:
                        predprob_techniques_corrected[i][k] = 0
    return predprob_techniques_corrected


def combinations(c, d):
    c_list = [c-0.1, c, c+0.1]
    d_list = [d-0.1, d, d+0.1]
    return [[cl, dl] for cl in c_list for dl in d_list]


def hanging_node_threshold_comparison(pred_tactics, predprob_tactics, pred_techniques, predprob_techniques, known_pred_techniques, permutations):
    f05list = []
    for pl in permutations:
        new_pred_techniques = hanging_node(pred_tactics, predprob_tactics, pred_techniques, predprob_techniques, pl[0], pl[1])
        f05list.append([pl, fbeta_score(known_pred_techniques, new_pred_techniques, beta=0.5, average='macro')])
    return f05list


def print_progress_bar(iteration):
    percent = ("{0:.1f}").format(100 * (iteration / float(50)))
    filledLength = int(iteration)
    bar = 'X' * filledLength + '-' * (50 - filledLength)
    print('\rProgress: |%s| %s%% Complete' % (bar, percent), end='\r')
    if iteration == 50:
        print()


def find_best_post_processing():
    stop_words = stopwords.words('english')
    new_stop_words = ["'ll", "'re", "'ve", 'ha', 'wa', "'d", "'s", 'abov', 'ani', 'becaus', 'befor', 'could', 'doe', 'dure', 'might', 'must', "n't", 'need', 'onc', 'onli', 'ourselv', 'sha', 'themselv', 'veri', 'whi', 'wo', 'would', 'yourselv']
    stop_words.extend(new_stop_words)

    train_data_df = pd.read_csv('./Entity-Extraction/rcATT/Dataset.csv', encoding="ISO-8859-1")
    train_data_df = processing(train_data_df)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reports = train_data_df[TEXT_FEATURES]
    overall_ttps = train_data_df[ALL_TTPS]

    parameters = joblib.load("./Entity-Extraction/rcATT/Models/configuration.joblib")
    c = parameters[1][0]
    d = parameters[1][1]
    permutations = combinations(c, d)

    f05_NO = []
    f05_HN = []
    f05_CP = []

    min_prob_tactics = 0.0
    max_prob_tactics = 0.0
    min_prob_techniques = 0.0
    max_prob_techniques = 0.0

    i = 6

    for index1, index2 in kf.split(reports, overall_ttps):
        reports_train, reports_test = reports.iloc[index1], reports.iloc[index2]
        overall_ttps_train, overall_ttps_test = overall_ttps.iloc[index1], overall_ttps.iloc[index2]

        train_reports = reports_train[TEXT_FEATURES]
        test_reports = reports_test[TEXT_FEATURES]
        train_tactics = overall_ttps_train[CODE_TACTICS]
        train_techniques = overall_ttps_train[CODE_TECHNIQUES]
        test_tactics = overall_ttps_test[CODE_TACTICS]
        test_techniques = overall_ttps_test[CODE_TECHNIQUES]

        pipeline_tactics = Pipeline([
            ('columnselector', TextSelector(key='processed')),
            ('tfidf', TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words, max_df=0.90)),
            ('selection', SelectPercentile(chi2, percentile=50)),
            ('classifier', OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=True, class_weight='balanced'), n_jobs=1))
        ])
        pipeline_tactics.fit(train_reports, train_tactics)
        pred_tactics = pipeline_tactics.predict(test_reports)
        predprob_tactics = pipeline_tactics.decision_function(test_reports)

        if np.amin(predprob_tactics) < min_prob_tactics:
            min_prob_tactics = np.amin(predprob_tactics)
        if np.amax(predprob_tactics) > max_prob_tactics:
            max_prob_tactics = np.amax(predprob_tactics)

        print_progress_bar(i)

        pipeline_techniques = Pipeline([
            ('columnselector', TextSelector(key='processed')),
            ('tfidf', TfidfVectorizer(tokenizer=StemTokenizer(), stop_words=stop_words, min_df=2, max_df=0.99)),
            ('selection', SelectPercentile(chi2, percentile=50)),
            ('classifier', OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=False, max_iter=1000, class_weight='balanced'), n_jobs=1))
        ])
        pipeline_techniques.fit(train_reports, train_techniques)
        pred_techniques = pipeline_techniques.predict(test_reports)
        predprob_techniques = pipeline_techniques.decision_function(test_reports)

        if np.amin(predprob_techniques) < min_prob_techniques:
            min_prob_techniques = np.amin(predprob_techniques)
        if np.amax(predprob_techniques) > max_prob_techniques:
            max_prob_techniques = np.amax(predprob_techniques)

        i += 2
        print_progress_bar(i)

        f05_NO.append(fbeta_score(test_techniques, pred_techniques, beta=0.5, average='macro'))
        f05_HN.extend(hanging_node_threshold_comparison(pred_tactics, predprob_tactics, pred_techniques, predprob_techniques, test_techniques, permutations))

        i += 2
        print_progress_bar(i)

        CPres, _ = confidence_propagation(predprob_tactics, pred_techniques, predprob_techniques)

        i += 2
        print_progress_bar(i)

        f05_CP.append(fbeta_score(test_techniques, CPres, beta=0.5, average='macro'))
        i += 2

    save_post_processing_comparison = []
    fb05_NO_avg = np.mean(f05_NO)
    fb05_CP_avg = np.mean(f05_CP)
    best_HN = []
    fb05_Max_HN_avg = 0

    print_progress_bar(48)

    for ps in permutations:
        sum_list = []
        for prhn in f05_HN:
            if ps == prhn[0]:
                sum_list.append(prhn[1])
        avg_temp = np.mean(sum_list)
        if avg_temp >= fb05_Max_HN_avg:
            fb05_Max_HN_avg = avg_temp
            best_HN = ps

    if fb05_NO_avg >= fb05_CP_avg and fb05_NO_avg >= fb05_Max_HN_avg:
        save_post_processing_comparison = ["N"]
    elif fb05_CP_avg >= fb05_Max_HN_avg and fb05_CP_avg >= fb05_NO_avg:
        save_post_processing_comparison = ["CP"]
    else:
        save_post_processing_comparison = ["HN"]
    save_post_processing_comparison.extend([best_HN, [min_prob_tactics, max_prob_tactics], [min_prob_techniques, max_prob_techniques]])

    joblib.dump(save_post_processing_comparison, "./Entity-Extraction/rcATT/Models/configuration.joblib")

    print_progress_bar(50)
    print()


def train():
    print('[1/3] Finding best post-processing method...')
    find_best_post_processing()

    print('[2/3] Training final models on full dataset...')

    stop_words = stopwords.words('english')
    new_stop_words = ["'ll", "'re", "'ve", 'ha', 'wa', "'d", "'s", 'abov', 'ani', 'becaus', 'befor', 'could', 'doe', 'dure', 'might', 'must', "n't", 'need', 'onc', 'onli', 'ourselv', 'sha', 'themselv', 'veri', 'whi', 'wo', 'would', 'yourselv']
    stop_words.extend(new_stop_words)

    train_data_df = pd.read_csv('./Entity-Extraction/rcATT/Dataset.csv', encoding="ISO-8859-1")
    train_data_df = processing(train_data_df)

    reports = train_data_df[TEXT_FEATURES]
    tactics = train_data_df[CODE_TACTICS]
    techniques = train_data_df[CODE_TECHNIQUES]

    pipeline_tactics = Pipeline([
        ('columnselector', TextSelector(key='processed')),
        ('tfidf', TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words, max_df=0.90)),
        ('selection', SelectPercentile(chi2, percentile=50)),
        ('classifier', OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=True, class_weight='balanced'), n_jobs=1))
    ])
    pipeline_tactics.fit(reports, tactics)
    print('  Tactics model trained.')

    pipeline_techniques = Pipeline([
        ('columnselector', TextSelector(key='processed')),
        ('tfidf', TfidfVectorizer(tokenizer=StemTokenizer(), stop_words=stop_words, min_df=2, max_df=0.99)),
        ('selection', SelectPercentile(chi2, percentile=50)),
        ('classifier', OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=False, max_iter=1000, class_weight='balanced'), n_jobs=1))
    ])
    pipeline_techniques.fit(reports, techniques)
    print('  Techniques model trained.')

    joblib.dump(pipeline_tactics, './Entity-Extraction/rcATT/Models/tactics.joblib')
    joblib.dump(pipeline_techniques, './Entity-Extraction/rcATT/Models/techniques.joblib')

    print('[3/3] Models saved to ./Entity-Extraction/rcATT/Models/')
    print('  - tactics.joblib')
    print('  - techniques.joblib')
    print('  - configuration.joblib')
    print('[DONE] Training complete!')


if __name__ == '__main__':
    train()
