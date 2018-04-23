import pandas as pd
import printerPlot
import parserExcel as parser
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import pickle


# Download dataset
db = parser.getDataFramefromExcel("./DATA2.csv")

features_prec_tard = db.columns[8:]
features_prec = db.columns[8:-6]
target = "bpd"

tr_set, ts_set = parser.splitFrom(db, "byear", 2015)
ts_set.reset_index(inplace=True, drop=True)

y_tr = tr_set[target].tolist()
y_ts = ts_set[target].tolist()


# Best Models
# Gradient Boosting Tree
gbc_prec_tard = pickle.load(open("XGBClassifier_precoci_tardardivi.p","rb"))
gbc_prec = pickle.load(open("XGBClassifier_precoci.p","rb"))

# Random Forest
rfc_prec_tard = pickle.load(open("RandomForestClassifier_precoci_tardivi.p","rb"))
rfc_prec = pickle.load(open("RandomForestClassifier_precoci.p","rb"))

# Decision Tree
dtc_prec_tard = pickle.load(open("DecisionTreeClassifier_precoci_tardivi.p","rb"))
dtc_prec = pickle.load(open("DecisionTreeClassifier_precoci.p","rb"))

# Logistic Regression
lrc_prec_tard = pickle.load(open("LogisticRegression_precoci_tardivi.p","rb"))
lrc_prec = pickle.load(open("LogisticRegression_precoci.p","rb"))


# Fit models
gbc_prec.fit(tr_set[features_prec], y_tr)
gbc_prec_tard.fit(tr_set[features_prec_tard], y_tr)

rfc_prec.fit(tr_set[features_prec], y_tr)
rfc_prec_tard.fit(tr_set[features_prec_tard], y_tr)

dtc_prec.fit(tr_set[features_prec], y_tr)
dtc_prec_tard.fit(tr_set[features_prec_tard], y_tr)

scaler = StandardScaler()
scaled_df = scaler.fit_transform(tr_set[features_prec])
x_tr = pd.DataFrame(scaled_df, columns=features_prec)
lrc_prec.fit(x_tr, y_tr)

scaled_df = scaler.fit_transform(tr_set[features_prec_tard])
x_tr = pd.DataFrame(scaled_df, columns=features_prec_tard)
lrc_prec_tard.fit(x_tr, y_tr)


# Write features importance and coefficent into file
with open("Features_importance_AND_coeff.txt", "w") as f:
    f.write("features_importance GB_prec_tard:  " + str(list(zip(features_prec_tard, gbc_prec_tard.feature_importances_))))
    f.write("\n\n")
    f.write("features_importance GB_prec:  " + str(list(zip(features_prec, gbc_prec.feature_importances_))))
    f.write("\n\n")
    f.write("features_importance RF_prec_tard:  " + str(list(zip(features_prec_tard, rfc_prec_tard.feature_importances_))))
    f.write("\n\n")
    f.write("features_importance RF_prec:  " + str(list(zip(features_prec, rfc_prec.feature_importances_))))
    f.write("\n\n")
    f.write("features_importance DT_prec_tard:  " + str(list(zip(features_prec_tard, dtc_prec_tard.feature_importances_))))
    f.write("\n\n")
    f.write("features_importance DT_prec:  " + str(list(zip(features_prec, dtc_prec.feature_importances_))))
    f.write("\n\n")
    f.write("coeficienti LR_prec_tard  " + str(lrc_prec_tard.coef_))
    f.write("\n\n")
    f.write("coeficienti LR_prec  " + str(lrc_prec.coef_))
f.close()

# Draw tree
printerPlot.visualize_tree(dtc_prec_tard, features_prec_tard, "DT_VON-PT")
printerPlot.visualize_tree(dtc_prec, features_prec, "DT_VON-P")
#printerTree.visualize_tree(gbc_prec_tard, features_prec_tard, "GB_VON-PT")
#printerTree.visualize_tree(gbc_prec, features_prec, "GB_VON-P")

# Draw features importance
printerPlot.histogram(rfc_prec_tard.feature_importances_, features_prec_tard, title="features_imp_RF_VON-PT")
printerPlot.histogram(rfc_prec.feature_importances_, features_prec, title="features_imp_RF_VON-P")
printerPlot.histogram(dtc_prec_tard.feature_importances_, features_prec_tard, title="features_imp_DT_VON-PT")
printerPlot.histogram(dtc_prec.feature_importances_, features_prec, title="features_imp_DT_VON-P")
printerPlot.histogram(gbc_prec_tard.feature_importances_, features_prec_tard, title="features_imp_GB_VON-PT")
printerPlot.histogram(gbc_prec.feature_importances_, features_prec, title="features_imp_GB_VON-P")


def scoring(res, y_true, res_proba, name):
    pos = 0
    i = 0
    for y in y_true:
        if res[i] == y:
            pos += 1
        i += 1
    cm = confusion_matrix(y_true, res)
    #printerTree.plot_confusion_matrix(cm,[0,1])

    with open("result.txt", "a") as f:
        f.write("\n\n")
        f.write(name)
        f.write("\n")
        f.write("Confusion matrix:\n" + str(cm))
        f.write("\n")
        f.write("Correct result: " + str(pos) + "/" + str(len(res)))
        f.write("\n")
        f.write("Classification f1_score is " + str(f1_score(y_true, res, average="micro")))
        f.write("\n")
        f.write("Classification roc_auc_score is " + str(roc_auc_score(y_true, res_proba)))
    f.close()


# Run test
gp = gbc_prec.predict(ts_set[features_prec])
gpt = gbc_prec_tard.predict(ts_set[features_prec_tard])

rp = rfc_prec.predict(ts_set[features_prec])
rpt = rfc_prec_tard.predict(ts_set[features_prec_tard])

dp = dtc_prec.predict(ts_set[features_prec])
dpt = dtc_prec_tard.predict(ts_set[features_prec_tard])

scaler = StandardScaler()
scaled_df = scaler.fit_transform(ts_set[features_prec])
x_ts = pd.DataFrame(scaled_df, columns=features_prec)
lp = lrc_prec.predict(x_ts)
lp_proba = lrc_prec.predict_proba(x_ts)[:, 1]

scaled_df = scaler.fit_transform(ts_set[features_prec_tard])
x_ts = pd.DataFrame(scaled_df, columns=features_prec_tard)
lpt = lrc_prec_tard.predict(x_ts)
lpt_proba = lrc_prec_tard.predict_proba(x_ts)[:, 1]

scoring(gp, y_ts, gbc_prec.predict_proba(ts_set[features_prec])[:, 1], "GB_prec")
scoring(gpt, y_ts, gbc_prec_tard.predict_proba(ts_set[features_prec_tard])[:, 1], "GB_prec_tard")

scoring(rp, y_ts, rfc_prec.predict_proba(ts_set[features_prec])[:, 1], "RF_prec")
scoring(rpt, y_ts, rfc_prec_tard.predict_proba(ts_set[features_prec_tard])[:, 1], "RF_prec_tard")

scoring(dp, y_ts, dtc_prec.predict_proba(ts_set[features_prec])[:, 1], "DT_prec")
scoring(dpt, y_ts, dtc_prec_tard.predict_proba(ts_set[features_prec_tard])[:, 1], "DT_prec_tard")

scoring(lp, y_ts, lp_proba, "LR_prec")
scoring(lpt, y_ts, lpt_proba, "LR_prec_tard")


# Check results similarities
# check different prediction
def different_prediction(ts_set, rp, dp, lp, rpt, dpt, lpt, gp, gpt, name=""):
    no_eq_rf_dt = []
    no_eq_rf_lr = []
    no_eq_dt_lr = []
    no_eq_dt_gb = []
    no_eq_rf_gb = []
    no_eq_lr_gb = []
    no_eq_rf_dt_tard = []
    no_eq_rf_lr_tard = []
    no_eq_dt_lr_tard = []
    no_eq_dt_gb_tard = []
    no_eq_rf_gb_tard = []
    no_eq_lr_gb_tard = []
    for i in range(0, len(rp)):
        if rp[i] != dp[i]:
            patient = {'hospno': ts_set.at[i,"hospno"], 'id': ts_set.at[i,"id"], 'rf':rp[i], "dt":dp[i]}
            no_eq_rf_dt.append(patient)
        if rp[i] != lp[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'rf': rp[i], "lr": lp[i]}
            no_eq_rf_lr.append(patient)
        if dp[i] != lp[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'dt': dp[i], "lr": lp[i]}
            no_eq_dt_lr.append(patient)
        if dp[i] != gp[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'dt': dp[i], "gb": gp[i]}
            no_eq_dt_gb.append(patient)
        if lp[i] != gp[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'lr': lp[i], "gb": gp[i]}
            no_eq_lr_gb.append(patient)
        if rp[i] != gp[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'rf': rp[i], "gb": gp[i]}
            no_eq_rf_gb.append(patient)

        if rpt[i] != dpt[i]:
            patient = {'hospno': ts_set.at[i,"hospno"], 'id': ts_set.at[i,"id"], 'rf':rpt[i], "dt":dpt[i]}
            no_eq_rf_dt_tard.append(patient)
        if rpt[i] != lpt[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'rf': rpt[i], "lr": lpt[i]}
            no_eq_rf_lr_tard.append(patient)
        if dpt[i] != lpt[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'dt': dpt[i], "lr": lpt[i]}
            no_eq_dt_lr_tard.append(patient)
        if dpt[i] != gpt[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'dt': dpt[i], "gb": gpt[i]}
            no_eq_dt_gb_tard.append(patient)
        if lpt[i] != gpt[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'lr': lpt[i], "gb": gpt[i]}
            no_eq_lr_gb_tard.append(patient)
        if rpt[i] != gpt[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'rf': rpt[i], "gb": gpt[i]}
            no_eq_rf_gb_tard.append(patient)

    diff = {"RF-DT":no_eq_rf_dt, "RF-LR":no_eq_rf_lr, "DT-LR":no_eq_dt_lr, "DT-GB":no_eq_dt_gb, "LR-GB":no_eq_lr_gb, "RF-GB":no_eq_rf_gb}
    diff_t = {"RF-DT":no_eq_rf_dt_tard, "RF-LR":no_eq_rf_lr_tard, "DT-LR":no_eq_dt_lr_tard, "DT-GB":no_eq_dt_gb_tard, "LR-GB":no_eq_lr_gb_tard, "RF-GB":no_eq_rf_gb_tard}
    pickle.dump({"prec":diff, "prec_tard":diff_t}, open("result_diff"+name+".p", "wb"))
    with open("result_differences"+name+".txt", "w") as f:
        f.write("Predizioni non uguali")
        f.write("\n")
        f.write("Differenze PRECOCI")
        f.write("\n")
        f.write("diff RF-LR= " + str(len(no_eq_rf_lr)))
        f.write("\n")
        f.write("diff RF-DT= "+str(len(no_eq_rf_dt)))
        f.write("\n")
        f.write("diff DT-LR= " + str(len(no_eq_dt_lr)))
        f.write("\n")
        f.write("diff RF-GB= " + str(len(no_eq_rf_gb)))
        f.write("\n")
        f.write("diff DT-GB= " + str(len(no_eq_dt_gb)))
        f.write("\n")
        f.write("diff LR-GB= " + str(len(no_eq_lr_gb)))
        f.write("\n\n")
        f.write("Differenze PRECOCI-TARDIVI")
        f.write("\n")
        f.write("diff RF-LR= " + str(len(no_eq_rf_lr_tard)))
        f.write("\n")
        f.write("diff RF-DT= " + str(len(no_eq_rf_dt_tard)))
        f.write("\n")
        f.write("diff DT-LR= " + str(len(no_eq_dt_lr_tard)))
        f.write("\n")
        f.write("diff RF-GB= " + str(len(no_eq_rf_gb_tard)))
        f.write("\n")
        f.write("diff DT-GB= " + str(len(no_eq_dt_gb_tard)))
        f.write("\n")
        f.write("diff LR-GB= " + str(len(no_eq_lr_gb_tard)))
        f.write("\n\n")
        f.write(str({"prec":diff, "prec_tard":diff_t}))
    f.close()


different_prediction(ts_set, rp, dp, lp, rpt, dpt, lpt, gp, gpt)


# check same wrong result
def same_wrong(ts_set, res, rp, dp, lp, rpt, dpt, lpt, gp, gpt, name=""):
    wrong_simil_rf_dt = []
    wrong_simil_rf_lr = []
    wrong_simil_dt_lr = []
    wrong_simil_rf_dt_tard = []
    wrong_simil_rf_lr_tard = []
    wrong_simil_dt_lr_tard = []
    wrong_simil_dt_gb = []
    wrong_simil_rf_gb = []
    wrong_simil_lr_gb = []
    wrong_simil_dt_gb_tard = []
    wrong_simil_rf_gb_tard = []
    wrong_simil_lr_gb_tard = []
    for i in range(0, len(rp)):
        if rp[i] != res[i]:
            patient = {'hospno': ts_set.at[i,"hospno"], 'id': ts_set.at[i,"id"], 'rf':rp[i], "res":res[i]}
            if rp[i] == dp[i]:
                wrong_simil_rf_dt.append(patient)
            if rp[i] == lp[i]:
                wrong_simil_rf_lr.append(patient)
            if rp[i] == gp[i]:
                wrong_simil_rf_gb.append(patient)

        if dp[i] != res[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'dt': dp[i], "res": res[i]}
            if dp[i] == lp[i]:
                wrong_simil_dt_lr.append(patient)
            if dp[i] == gp[i]:
                wrong_simil_dt_gb.append(patient)
        if gp[i] != res[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'gb': gp[i], "res": res[i]}
            if gp[i] == lp[i]:
                wrong_simil_lr_gb.append(patient)

        if rpt[i] != res[i]:
            patient = {'hospno': ts_set.at[i,"hospno"], 'id': ts_set.at[i,"id"], 'rf':rpt[i], "res":res[i]}
            if rpt[i] == dpt[i]:
                wrong_simil_rf_dt_tard.append(patient)
            if rpt[i] == lpt[i]:
                wrong_simil_rf_lr_tard.append(patient)
            if rpt[i] == gpt[i]:
                wrong_simil_rf_gb_tard.append(patient)
        if dpt[i] != res[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'dt': dpt[i], "res": res[i]}
            if dpt[i] == lpt[i]:
                wrong_simil_dt_lr_tard.append(patient)
            if dpt[i] == gpt[i]:
                wrong_simil_dt_gb_tard.append(patient)
        if gpt[i] != res[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'gb': gpt[i], "res": res[i]}
            if gpt[i] == lpt[i]:
                wrong_simil_lr_gb_tard.append(patient)

    diff = {"RF-DT":wrong_simil_rf_dt, "RF-LR":wrong_simil_rf_lr, "DT-LR":wrong_simil_dt_lr, "DT-GB":wrong_simil_dt_gb, "LR-GB":wrong_simil_lr_gb, "RF-GB":wrong_simil_rf_gb}
    diff_t = {"RF-DT":wrong_simil_rf_dt_tard, "RF-LR":wrong_simil_rf_lr_tard, "DT-LR":wrong_simil_dt_lr_tard, "DT-GB":wrong_simil_dt_gb_tard, "LR-GB":wrong_simil_lr_gb_tard, "RF-GB":wrong_simil_rf_gb_tard}
    pickle.dump({"prec":diff, "prec_tard":diff_t}, open("result_wrong"+name+".p", "wb"))
    with open("result_same_wrong"+name+".txt", "w") as f:
        f.write("Predizioni degli stessi sbagliati")
        f.write("\n")
        f.write("Differenze PRECOCI")
        f.write("\n")
        f.write("diff RF-LR= " + str(len(wrong_simil_rf_lr)))
        f.write("\n")
        f.write("diff RF-DT= "+str(len(wrong_simil_rf_dt)))
        f.write("\n")
        f.write("diff DT-LR= " + str(len(wrong_simil_dt_lr)))
        f.write("\n")
        f.write("diff RF-GB= " + str(len(wrong_simil_rf_gb)))
        f.write("\n")
        f.write("diff DT-GB= " + str(len(wrong_simil_dt_gb)))
        f.write("\n")
        f.write("diff LR-GB= " + str(len(wrong_simil_lr_gb)))
        f.write("\n\n")
        f.write("Differenze PRECOCI-TARDIVI")
        f.write("\n")
        f.write("diff RF-LR= " + str(len(wrong_simil_rf_lr_tard)))
        f.write("\n")
        f.write("diff RF-DT= " + str(len(wrong_simil_rf_dt_tard)))
        f.write("\n")
        f.write("diff DT-LR= " + str(len(wrong_simil_dt_lr_tard)))
        f.write("\n")
        f.write("diff RF-GB= " + str(len(wrong_simil_rf_gb_tard)))
        f.write("\n")
        f.write("diff DT-GB= " + str(len(wrong_simil_dt_gb_tard)))
        f.write("\n")
        f.write("diff LR-GB= " + str(len(wrong_simil_lr_gb_tard)))
        f.write("\n\n")
        f.write(str({"prec":diff, "prec_tard":diff_t}))
    f.close()

same_wrong(ts_set, y_ts, rp, dp, lp, rpt, dpt, lpt, gp, gpt)


# check same correct result
def same_correct(ts_set, res, rp, dp, lp, rpt, dpt, lpt, gp, gpt, name=""):
    correct_rf_dt = []
    correct_rf_lr = []
    correct_dt_lr = []
    correct_rf_dt_tard = []
    correct_rf_lr_tard = []
    correct_dt_lr_tard = []
    correct_dt_gb = []
    correct_rf_gb = []
    correct_lr_gb = []
    correct_dt_gb_tard = []
    correct_rf_gb_tard = []
    correct_lr_gb_tard = []
    for i in range(0, len(rp)):
        if rp[i] == res[i]:
            patient = {'hospno': ts_set.at[i,"hospno"], 'id': ts_set.at[i,"id"], 'rf':rp[i], "res":res[i]}
            if rp[i] == dp[i]:
                correct_rf_dt.append(patient)
            if rp[i] == lp[i]:
                correct_rf_lr.append(patient)
            if rp[i] == gp[i]:
                correct_rf_gb.append(patient)
        if dp[i] == res[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'dt': dp[i], "res": res[i]}
            if dp[i] == lp[i]:
                correct_dt_lr.append(patient)
            if dp[i] == gp[i]:
                correct_dt_gb.append(patient)
        if gp[i] == res[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'gb': gp[i], "res": res[i]}
            if gp[i] == lp[i]:
                correct_lr_gb.append(patient)

        if rpt[i] == res[i]:
            patient = {'hospno': ts_set.at[i,"hospno"], 'id': ts_set.at[i,"id"], 'rf':rpt[i], "res":res[i]}
            if rpt[i] == dpt[i]:
                correct_rf_dt_tard.append(patient)
            if rpt[i] == lpt[i]:
                correct_rf_lr_tard.append(patient)
            if rpt[i] == gpt[i]:
                correct_rf_gb_tard.append(patient)
        if dpt[i] == res[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'dt': dpt[i], "res": res[i]}
            if dpt[i] == lpt[i]:
                correct_dt_lr_tard.append(patient)
            if dpt[i] == gpt[i]:
                correct_dt_gb_tard.append(patient)
        if gpt[i] == res[i]:
            patient = {'hospno': ts_set.at[i, "hospno"], 'id': ts_set.at[i, "id"], 'gb': gpt[i], "res": res[i]}
            if gpt[i] == lpt[i]:
                correct_lr_gb_tard.append(patient)

    diff = {"RF-DT":correct_rf_dt, "RF-LR":correct_rf_lr, "DT-LR":correct_dt_lr, "DT-GB":correct_dt_gb, "LR-GB":correct_lr_gb, "RF-GB":correct_rf_gb}
    diff_t = {"RF-DT":correct_rf_dt_tard, "RF-LR":correct_rf_lr_tard, "DT-LR":correct_dt_lr_tard, "DT-GB":correct_dt_gb_tard, "LR-GB":correct_lr_gb_tard, "RF-GB":correct_rf_gb_tard}
    pickle.dump({"prec":diff, "prec_tard":diff_t}, open("result_correct"+name+".p", "wb"))
    with open("result_same_correct"+name+".txt", "w") as f:
        f.write("Predizioni degli stessi corretti")
        f.write("\n")
        f.write("Differenze PRECOCI")
        f.write("\n")
        f.write("diff RF-LR= " + str(len(correct_rf_lr)))
        f.write("\n")
        f.write("diff RF-DT= "+str(len(correct_rf_dt)))
        f.write("\n")
        f.write("diff DT-LR= " + str(len(correct_dt_lr)))
        f.write("\n")
        f.write("diff RF-GB= " + str(len(correct_rf_gb)))
        f.write("\n")
        f.write("diff DT-GB= " + str(len(correct_dt_gb)))
        f.write("\n")
        f.write("diff LR-GB= " + str(len(correct_lr_gb)))
        f.write("\n\n")
        f.write("Differenze PRECOCI-TARDIVI")
        f.write("\n")
        f.write("diff RF-LR= " + str(len(correct_rf_lr_tard)))
        f.write("\n")
        f.write("diff RF-DT= " + str(len(correct_rf_dt_tard)))
        f.write("\n")
        f.write("diff DT-LR= " + str(len(correct_dt_lr_tard)))
        f.write("\n")
        f.write("diff RF-GB= " + str(len(correct_rf_gb_tard)))
        f.write("\n")
        f.write("diff DT-GB= " + str(len(correct_dt_gb_tard)))
        f.write("\n")
        f.write("diff LR-GB= " + str(len(correct_lr_gb_tard)))
        f.write("\n\n")
        f.write(str({"prec":diff, "prec_tard":diff_t}))
    f.close()

same_correct(ts_set, y_ts, rp, dp, lp, rpt, dpt, lpt, gp, gpt)
