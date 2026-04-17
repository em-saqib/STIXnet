import pandas as pd
import json
import csv
import sys
import os

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

CODE_TACTICS = ['TA0043', 'TA0042', 'TA0001', 'TA0002', 'TA0003', 'TA0004', 'TA0005', 'TA0006', 'TA0007', 'TA0008', 'TA0009', 'TA0011', 'TA0010', 'TA0040']

CODE_TECHNIQUES = ['T1595', 'T1592', 'T1589', 'T1590', 'T1591', 'T1598', 'T1597', 'T1596', 'T1593', 'T1594', 'T1583', 'T1586', 'T1584', 'T1587', 'T1585', 'T1588', 'T1608', 'T1189', 'T1190', 'T1133', 'T1200', 'T1566', 'T1091', 'T1195', 'T1199', 'T1078', 'T1059', 'T1609', 'T1610', 'T1203', 'T1559', 'T1106', 'T1053', 'T1129', 'T1072', 'T1569', 'T1204', 'T1047', 'T1098', 'T1197', 'T1547', 'T1037', 'T1176', 'T1554', 'T1136', 'T1543', 'T1546', 'T1574', 'T1525', 'T1556', 'T1137', 'T1542', 'T1505', 'T1205', 'T1548', 'T1134', 'T1484', 'T1611', 'T1068', 'T1055', 'T1612', 'T1622', 'T1140', 'T1006', 'T1480', 'T1211', 'T1222', 'T1564', 'T1562', 'T1070', 'T1202', 'T1036', 'T1578', 'T1112', 'T1601', 'T1599', 'T1027', 'T1647', 'T1620', 'T1207', 'T1014', 'T1553', 'T1218', 'T1216', 'T1221', 'T1127', 'T1535', 'T1550', 'T1497', 'T1600', 'T1220', 'T1557', 'T1110', 'T1555', 'T1212', 'T1187', 'T1606', 'T1056', 'T1111', 'T1621', 'T1040', 'T1003', 'T1528', 'T1558', 'T1539', 'T1552', 'T1087', 'T1010', 'T1217', 'T1580', 'T1538', 'T1526', 'T1619', 'T1613', 'T1482', 'T1083', 'T1615', 'T1046', 'T1135', 'T1201', 'T1120', 'T1069', 'T1057', 'T1012', 'T1018', 'T1518', 'T1082', 'T1614', 'T1016', 'T1049', 'T1033', 'T1007', 'T1124', 'T1210', 'T1534', 'T1570', 'T1563', 'T1021', 'T1080', 'T1560', 'T1123', 'T1119', 'T1185', 'T1115', 'T1530', 'T1602', 'T1213', 'T1005', 'T1039', 'T1025', 'T1074', 'T1114', 'T1113', 'T1125', 'T1071', 'T1092', 'T1132', 'T1001', 'T1568', 'T1573', 'T1008', 'T1105', 'T1104', 'T1095', 'T1571', 'T1572', 'T1090', 'T1219', 'T1102', 'T1020', 'T1030', 'T1048', 'T1041', 'T1011', 'T1052', 'T1567', 'T1029', 'T1537', 'T1531', 'T1485', 'T1486', 'T1565', 'T1491', 'T1561', 'T1499', 'T1495', 'T1490', 'T1498', 'T1496', 'T1489', 'T1529']

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

# Load from local enterprise-attack.json instead of API
local_json = os.path.join(os.path.dirname(__file__), 'enterprise-attack.json')
print(f'[Loading local MITRE ATT&CK data from {local_json}]')
with open(local_json, 'r') as f:
    attack_data = json.load(f)

all_objects = attack_data['objects']
tactics = [o for o in all_objects if o.get('type') == 'x-mitre-tactic' and not o.get('revoked', False)]
techniques = [o for o in all_objects if o.get('type') == 'attack-pattern' and not o.get('revoked', False)]

print(f'[Loaded {len(tactics)} tactics, {len(techniques)} techniques]')

header = ['Text']
for code in CODE_TACTICS:
    header.append(code)
for code in CODE_TECHNIQUES:
    header.append(code)

n = len(header) - 1

with open('./Entity-Extraction/rcATT/Dataset.csv', 'w', encoding="utf8") as dataset:
    writer = csv.writer(dataset, lineterminator='\n')
    writer.writerow(header)

    i_tact = 0
    print('[WRITING TACTICS]')
    for code in CODE_TACTICS:
        for tact in tactics:
            if tact['external_references'][0]['external_id'] == code:
                row = []
                row.append(tact.get('description', '').replace("\n", " "))
                listofzeros = ['0'] * n
                listofzeros[i_tact] = '1'
                i_tact += 1
                for item in listofzeros:
                    row.append(item)
                writer.writerow(row)

    print('[WRITING TECHNIQUES]')
    for code in CODE_TECHNIQUES:
        for tech in techniques:
            if tech['external_references'][0]['external_id'].split('.')[0] == code:
                row = []
                row.append(tech.get('description', '').replace("\n", " "))
                index = 15 + CODE_TECHNIQUES.index(code)
                listofzeros = ['0'] * n
                listofzeros[index - 1] = '1'
                for code_tact in CODE_TACTICS:
                    for code_tech in TACTICS_TECHNIQUES_RELATIONSHIP_DF[code_tact]:
                        if code_tech == code:
                            listofzeros[CODE_TACTICS.index(code_tact)] = '1'
                for item in listofzeros:
                    row.append(item)
                writer.writerow(row)

                # Check if there is a url folder for that
                technique_name = tech['name'].replace("/", "_")
                url_dir_path = './Entity-Extraction/rcATT/Dataset-Creation/URL_Content/' + technique_name + '/'
                if os.path.isdir(url_dir_path):
                    for url_file in os.listdir(url_dir_path):
                        url_row = []
                        url_path = url_dir_path + url_file
                        with open(url_path, 'r', encoding="utf8") as f:
                            content = f.read()
                            content = content.replace("\n", " ")
                            url_row.append(content)
                            for item in listofzeros:
                                url_row.append(item)
                            writer.writerow(url_row)

    # Skip oldDataset.csv if not present
    old_dataset_path = './Entity-Extraction/rcATT/Dataset-Creation/oldDataset.csv'
    if os.path.exists(old_dataset_path):
        print('[ADDING rcATT DATASET]')
        with open(old_dataset_path, 'r', encoding="utf8") as rcATT:
            reader = csv.reader(rcATT)
            rcATT_header = next(reader)
            new_ttp_header = header[1:]
            old_ttp_header = rcATT_header[1:]
            old_ttp = [h for h in old_ttp_header if h not in new_ttp_header]

            for i, line in enumerate(reader):
                rcATT_row = []
                text = line[0]
                old_ttp_indeces = line[1:]
                old_indeces = [i for i, j in enumerate(old_ttp_indeces) if '1' in j.lower()]
                new_ttp_indeces = ['0'] * n
                for old_index in old_indeces:
                    ttp_name = old_ttp_header[old_index]
                    if ttp_name not in old_ttp:
                        new_index = new_ttp_header.index(ttp_name)
                        new_ttp_indeces[new_index] = '1'
                rcATT_row.append(text)
                for item in new_ttp_indeces:
                    rcATT_row.append(item)
                writer.writerow(rcATT_row)
    else:
        print('[SKIPPING oldDataset.csv - file not found]')

print('[DONE] Dataset.csv created at ./Entity-Extraction/rcATT/Dataset.csv')
