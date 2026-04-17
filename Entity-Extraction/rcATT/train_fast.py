"""
Fast training script for rcATT - skips cross-validation, trains final models directly.
Run from: cd /home/saqib/cti/STIXnet && python3 Entity-Extraction/rcATT/train_fast.py
"""
import joblib
import pandas as pd
import re

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.base import BaseEstimator, TransformerMixin

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

TEXT_FEATURES = ['processed']

CODE_TACTICS = ['TA0043', 'TA0042', 'TA0001', 'TA0002', 'TA0003', 'TA0004', 'TA0005', 'TA0006', 'TA0007', 'TA0008', 'TA0009', 'TA0011', 'TA0010', 'TA0040']
CODE_TECHNIQUES = ['T1595', 'T1592', 'T1589', 'T1590', 'T1591', 'T1598', 'T1597', 'T1596', 'T1593', 'T1594', 'T1583', 'T1586', 'T1584', 'T1587', 'T1585', 'T1588', 'T1608', 'T1189', 'T1190', 'T1133', 'T1200', 'T1566', 'T1091', 'T1195', 'T1199', 'T1078', 'T1059', 'T1609', 'T1610', 'T1203', 'T1559', 'T1106', 'T1053', 'T1129', 'T1072', 'T1569', 'T1204', 'T1047', 'T1098', 'T1197', 'T1547', 'T1037', 'T1176', 'T1554', 'T1136', 'T1543', 'T1546', 'T1574', 'T1525', 'T1556', 'T1137', 'T1542', 'T1505', 'T1205', 'T1548', 'T1134', 'T1484', 'T1611', 'T1068', 'T1055', 'T1612', 'T1622', 'T1140', 'T1006', 'T1480', 'T1211', 'T1222', 'T1564', 'T1562', 'T1070', 'T1202', 'T1036', 'T1578', 'T1112', 'T1601', 'T1599', 'T1027', 'T1647', 'T1620', 'T1207', 'T1014', 'T1553', 'T1218', 'T1216', 'T1221', 'T1127', 'T1535', 'T1550', 'T1497', 'T1600', 'T1220', 'T1557', 'T1110', 'T1555', 'T1212', 'T1187', 'T1606', 'T1056', 'T1111', 'T1621', 'T1040', 'T1003', 'T1528', 'T1558', 'T1539', 'T1552', 'T1087', 'T1010', 'T1217', 'T1580', 'T1538', 'T1526', 'T1619', 'T1613', 'T1482', 'T1083', 'T1615', 'T1046', 'T1135', 'T1201', 'T1120', 'T1069', 'T1057', 'T1012', 'T1018', 'T1518', 'T1082', 'T1614', 'T1016', 'T1049', 'T1033', 'T1007', 'T1124', 'T1210', 'T1534', 'T1570', 'T1563', 'T1021', 'T1080', 'T1560', 'T1123', 'T1119', 'T1185', 'T1115', 'T1530', 'T1602', 'T1213', 'T1005', 'T1039', 'T1025', 'T1074', 'T1114', 'T1113', 'T1125', 'T1071', 'T1092', 'T1132', 'T1001', 'T1568', 'T1573', 'T1008', 'T1105', 'T1104', 'T1095', 'T1571', 'T1572', 'T1090', 'T1219', 'T1102', 'T1020', 'T1030', 'T1048', 'T1041', 'T1011', 'T1052', 'T1567', 'T1029', 'T1537', 'T1531', 'T1485', 'T1486', 'T1565', 'T1491', 'T1561', 'T1499', 'T1495', 'T1490', 'T1498', 'T1496', 'T1489', 'T1529']


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
    stop_words = stopwords.words('english')
    stop_words.extend(["'ll", "'re", "'ve", 'ha', 'wa', "'d", "'s"])

    print('[1/4] Loading dataset...')
    df = pd.read_csv('./Entity-Extraction/rcATT/Dataset.csv', encoding='ISO-8859-1')
    df = processing(df)
    print(f'  {len(df)} samples loaded.')

    reports = df[TEXT_FEATURES]
    tactics = df[CODE_TACTICS]
    techniques = df[CODE_TECHNIQUES]

    print('[2/4] Training tactics model...')
    pipeline_tactics = Pipeline([
        ('columnselector', TextSelector(key='processed')),
        ('tfidf', TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words, max_df=0.90)),
        ('selection', SelectPercentile(chi2, percentile=50)),
        ('classifier', OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=True, class_weight='balanced'), n_jobs=1))
    ])
    pipeline_tactics.fit(reports, tactics)
    print('  Done.')

    print('[3/4] Training techniques model...')
    pipeline_techniques = Pipeline([
        ('columnselector', TextSelector(key='processed')),
        ('tfidf', TfidfVectorizer(tokenizer=StemTokenizer(), stop_words=stop_words, min_df=2, max_df=0.99)),
        ('selection', SelectPercentile(chi2, percentile=50)),
        ('classifier', OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=False, max_iter=1000, class_weight='balanced'), n_jobs=1))
    ])
    pipeline_techniques.fit(reports, techniques)
    print('  Done.')

    print('[4/4] Saving models...')
    joblib.dump(pipeline_tactics, './Entity-Extraction/rcATT/Models/tactics.joblib')
    joblib.dump(pipeline_techniques, './Entity-Extraction/rcATT/Models/techniques.joblib')
    print('  Saved to ./Entity-Extraction/rcATT/Models/')
    print('    - tactics.joblib')
    print('    - techniques.joblib')
    print('[DONE] Training complete!')
