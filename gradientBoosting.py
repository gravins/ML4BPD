import pandas as pd
import parserExcel as parser
import xgboost as xgb
import classification as csf


db = parser.getDataFramefromExcel("./BPDdataset.csv")

# Remodel dataset
# Order columns like
# other, outcome, possible early risk factors, possible late risk factors
db = parser.orderColumns(db)


# Drop some ininfluent columns
db = db.drop(["die12", "zbw", "sga10", "locate", "pcare", "ox36", "vent36", "hfv36", "hfnc36", "nimv36", "cpap36", "sterbpd", "anyvent36w"], 1)
# Remove all rows where died==1 because they can't have bpd
db = db.drop(db[db["died"] == 1].index).reset_index(drop=True)

# Drop all rows with null value
for c in db.columns:
    db = db.drop(db[pd.isnull(db[c])].index).reset_index(drop=True)

# Drop gadays and insert into gaweeks
db["gaweeks"] = db["gaweeks"].astype(float) # cast entire column to float
for i in range(0, len(db)):
    db.at[i, "gaweeks"] = round(db.at[i, "gaweeks"] + (db.at[i, "gadays"] / 7), 4) # round to 4 decimals
db = db.drop(["gadays"], 1)

# Possible value for BPD: 0= no, 1= yes, 99= N/D; 7= unknown; 9= missing
# for this reason drop 99,9,7
l = len(db)
db = db.drop(db[db["bpd"] == 99].index).reset_index(drop=True)
db = db.drop(db[db["bpd"] == 9].index).reset_index(drop=True)
db = db.drop(db[db["bpd"] == 7].index).reset_index(drop=True)
#print(str(l-len(db))+" element was dropped beacuse bpd == 99 or 9 or 7 ( = N/D, unk, miss)")

# one hot encoding of race column
db["race_1"] = 0
db["race_2"] = 0
db["race_3"] = 0
db["race_4"] = 0
db["race_7"] = 0
for i in range(0, len(db)):
    col = "race_"+str(int(db.at[i, "race"]))
    db.at[i, col] = 1

# move "race_ " colunms instead of "race" and drop it
new_order = db.columns.tolist()
for i in range(1, 5):
    new_order.remove("race_"+str(i))
    new_order.insert(new_order.index("race"), "race_"+str(i))
new_order.remove("race_7")
new_order.insert(new_order.index("race"), "race_7")

db = db[new_order]
db = db.drop(["race"], 1)

# Features is all columns in possible early risk factors and possible late risk factors
features_prec_tard = []
notardive = []
tard = True
for c in range(db.columns.get_loc("drsurf"), len(db.columns)):
    features_prec_tard.append(db.columns[c])
    if "drcpap" in db.columns[c]:
        tard = False
    if tard:
        notardive.append(db.columns[c])

features_prec = []
for c in range(db.columns.get_loc('drsurf'), db.columns.get_loc('drcpap')+1):
    features_prec.append(db.columns[c])

target = "bpd"

# Split data in training and test set
# consider all data from 2015-2016 as test set
tr_set, ts_set = parser.splitFrom(db, "byear", 2015)

training_pt = tr_set[features_prec_tard]
training_p = tr_set[features_prec]
tr_target = tr_set[target]

ts_set.reset_index(inplace=True, drop=True)
test_pt = ts_set[features_prec_tard]
test_p = ts_set[features_prec]
ts_target = ts_set[target]

# create XGBoost version of tr_set and ts_set
xgtrain_pt = xgb.DMatrix(training_pt.values, tr_target.values)
xgtrain_p = xgb.DMatrix(training_p.values, tr_target.values)

xgtest_pt = xgb.DMatrix(test_pt.values)
xgtest_p = xgb.DMatrix(test_p.values)

gbc = xgb.XGBClassifier(n_job=-1, random_state=42, tree_method="auto")
param_grid_gb = {"n_estimators": [200, 300, 400],
                 "learning_rate": [1.00000000e-05,   1.00000000e-04,   1.00000000e-03, 1.00000000e-02,   1.00000000e-01, 1.00000000e+00],
                 "max_depth": [0, 3, 7, 13],# 0 indica nessun limite
                 "subsample": [0.5, 0.6, 0.7, 0.8, 0.9]
                 }


name = "precoci_tardivi"
n_job = -1
a = csf.classification(gbc, param_grid_gb, training_pt, tr_target.tolist(), test_pt, ts_target.tolist(), name, valid=True, n_job=n_job)

name = "precoci"
a = csf.classification(gbc, param_grid_gb, training_p, tr_target.tolist(), test_p, ts_target.tolist(), name, valid=True, n_job=n_job)

