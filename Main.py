import pandas as pd
from threading import Thread
import parserExcel as parser
import classification as csf
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
from time import time

def main():
    if len(sys.argv) != 2:
        raise Warning("\n\tYou must pass one and only one arguments now, this buggy version of Scikit-Learn pop from empty warning list.\n\tPlease specify the model to use:\n\t\tLR for logistic regression\n\t\tDT for decision tree\n\t\tRF for random forest\n\t\tGB for gradient boosting tree")
        exit(1)
    if not ("DT" in sys.argv[1] or "RF" in sys.argv[1] or "LR" in sys.argv[1] or "GB" in sys.argv[1]):
        raise Warning("Please specify the model to use:\n\t\tLR for logistic regression\n\t\tDT for decision tree\n\t\tRF for random forest\n\t\tGB for gradient boosting tree\n\t\tSVM for support vector machine")
        exit(1)

    start = time()

    db = parser.getDataFramefromExcel("./BPDdataset.csv")

    '''
    # Save stats of data into DataFrame
    stats = db.describe()
    # Add variance
    stats.loc["variance"] = [stats[col]["std"]*stats[col]["std"] for col in stats.columns]
    # Print on file
    parser.saveDataFrameintoExcel(stats, "./statsdb.csv")'''

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
    db["gaweeks"] = db["gaweeks"].asclassifier_type(float) # cast entire column to float
    for i in range(0, len(db)):
        db.at[i, "gaweeks"] = round(db.at[i, "gaweeks"] + (db.at[i, "gadays"] / 7), 4) # round to 4 decimals
    db = db.drop(["gadays"], 1)

    # Possible value for BPD: 0= no, 1= yes, 99= N/D; 7= unknown; 9= missing
    # for this reason drop 99,9,7
    l = len(db)
    db = db.drop(db[db["bpd"] == 99].index).reset_index(drop=True)
    db = db.drop(db[db["bpd"] == 9].index).reset_index(drop=True)
    db = db.drop(db[db["bpd"] == 7].index).reset_index(drop=True)

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

    # Save stats of data into DataFrame
    stats = db.describe()
    # Add variance
    stats.loc["variance"] = [stats[col]["std"] * stats[col]["std"] for col in stats.columns]
    # Print on file
    parser.saveDataFrameintoExcel(stats, "./STATSdb_afterRemodel.csv")

    # Print dataframe into excel file
    parser.saveDataFrameintoExcel(db, "./DB_completo_dopoRemodel.csv")

    # Features is all columns in possible early risk factors and possible late risk factors
    features_precoci_tardivi = []
    notardive = []
    tard = True
    for c in range(db.columns.get_loc("drsurf"), len(db.columns)):
        features_precoci_tardivi.append(db.columns[c])
        if "drcpap" in db.columns[c]:
            tard = False
        if tard:
            notardive.append(db.columns[c])

    features_precoci = []
    for c in range(db.columns.get_loc('drsurf'), db.columns.get_loc('drcpap')+1):
        features_precoci.append(db.columns[c])

    target = "bpd"

    # Print dataframe with all possible early/late risk factors into excel file
    parser.saveDataFrameintoExcel(db, "./DATA2.csv")

    # Print dataframe with no possible late risk factors into excel file
    parser.saveDataFrameintoExcel(db[notardive], "./DATA1.csv")

    # Split data in training and test set
    # consider all data from 2015-2016 as test set
    tr_set, ts_set = parser.splitFrom(db, "byear", 2015)


    def threaded_function(classifier_type, features, name, n_job=-1):
        x_tr = tr_set[features]
        y_tr = tr_set[target].tolist()
        x_ts = ts_set[features]
        y_ts = ts_set[target].tolist()

        if "RF" in classifier_type:
            rfc = RandomForestClassifier(n_jobs=n_job, random_state=42)
            # dict of params for grid search
            param_grid_rf = {"n_estimators": [10, 500, 1000],
                             "criterion": ["gini", "entropy"],
                             "max_depth": [None, 3, 8, 13],
                             "min_samples_split": [2, 10, 30],
                             "min_samples_leaf": [1, 10, 20],
                             "class_weight": ["balanced", None]
                             }

            if "tard" in name:
                #rfc = csf.classification(rfc, param_grid_rf, x_tr, y_tr, x_ts, y_ts, name, valid=0.20)
                # __classification return a dictionary
                res = csf.classification(rfc, param_grid_rf, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]

            else:
                res = csf.classification(rfc, param_grid_rf, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]

        elif "DT" in classifier_type:
            dtc = DecisionTreeClassifier(random_state=42)
            # dict of params for grid search
            param_grid_dtc = {"criterion": ["gini", "entropy"],
                              "max_depth": [None, 3, 8, 13],
                              "min_samples_split": [2, 10, 30],
                              "min_samples_leaf": [1, 10, 20],
                              "class_weight": ["balanced", None],
                              "splitter": ["best", "random"]
                              }

            if "tard" in name:
                # r = csf.classification(dtc, param_grid_dtc, x_tr, y_tr, x_ts, y_ts, name, valid=0.20)
                # __classification return a dictionary
                res = csf.classification(dtc, param_grid_dtc, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]
            else:
                res = csf.classification(dtc, param_grid_dtc, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]
        elif "LR" in classifier_type:
            # Normalize all variable
            scaler = StandardScaler()
            scaled_df = scaler.fit_transform(x_tr)
            x_tr = pd.DataFrame(scaled_df, columns=x_tr.columns)

            lr = LogisticRegression()

            # dict of params for grid search
            param_grid_lr = {"penalty": ["l1", "l2"],
                             "C": np.logspace(-5, 4, 10)
                             }

            if "tard" in name:
                # r = csf.classification(lr, param_grid_lr, x_tr, y_tr, x_ts, y_ts, name, valid=0.20)
                # __classification return a dictionary
                res = csf.classification(lr, param_grid_lr, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]
            else:
                res = csf.classification(lr, param_grid_lr, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]
        elif "GB" in classifier_type:
            gbc = GradientBoostingClassifier(random_state=42)
            # dict of params for grid search
            param_grid_gb = {"loss": ["deviance", "exponential"],
                             "n_estimators": [10, 100, 500],
                             "learning_rate": np.logspace(-5, 4, 10),
                             "criterion": ["friedman_mse", "mae"],
                             "max_depth": [None, 3, 7, 13],
                             "subsample": [0.5, 0.6, 0.7, 0.8, 0.9]
                             }

            if "tard" in name:
                # gbc = csf.classification(rfc, param_grid_rf, x_tr, y_tr, x_ts, y_ts, name, valid=0.20)
                # __classification return a dictionary
                res = csf.classification(gbc, param_grid_gb, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]
            else:
                res = csf.classification(gbc, param_grid_gb, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]
        elif "SVM" in classifier_type:
            gbc = SVC(random_state=42)
            # dict of params for grid search
            param_grid_gb = {"C": [0.001, 0.1, 0.5, 1.0],
                "gamma":["auto"],
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "degree": [1, 3, 4],
                "shrinking": [True, False]
                }

            if "tard" in name:
                # gbc = csf.classification(rfc, param_grid_rf, x_tr, y_tr, x_ts, y_ts, name, valid=0.20)
                # __classification return a dictionary
                res = csf.classification(gbc, param_grid_gb, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]
            else:
                res = csf.classification(gbc, param_grid_gb, x_tr, y_tr, x_ts, y_ts, name, valid=True, n_job=n_job)["valid"]

        print(res)

    if "LR" in sys.argv[1]:
        t2_LR = Thread(target=threaded_function, args=("LR", features_precoci, "precoci"))
        t1_LR = Thread(target=threaded_function, args=("LR", features_precoci_tardivi, "precoci_tardivi"))
        t2_LR.start()
        t1_LR.start()
        t2_LR.join()
        t1_LR.join()
    elif "DT" in sys.argv[1]:
        t2_DT = Thread(target=threaded_function, args=("DT", features_precoci, "precoci"))
        t1_DT = Thread(target=threaded_function, args=("DT", features_precoci_tardivi, "precoci_tardivi"))
        t2_DT.start()
        t1_DT.start()
        t2_DT.join()
        t1_DT.join()
    elif "RF" in sys.argv[1]:
        t2_RF = Thread(target=threaded_function, args=("RF", features_precoci, "precoci"))
        t1_RF = Thread(target=threaded_function, args=("RF", features_precoci_tardivi, "precoci_tardivi"))
        t2_RF.start()
        t1_RF.start()
        t2_RF.join()
        t1_RF.join()
    elif "GB" in sys.argv[1]:
        t2_GB = Thread(target=threaded_function, args=("GB", features_precoci, "precoci"))
        t1_GB = Thread(target=threaded_function, args=("GB", features_precoci_tardivi, "precoci_tardivi"))
        t2_GB.start()
        t1_GB.start()
        t2_GB.join()
        t1_GB.join()
    elif "SVM" in sys.argv[1]:
        t2_SVM = Thread(target=threaded_function, args=("SVM", features_precoci, "precoci"))
        t1_SVM = Thread(target=threaded_function, args=("SVM", features_precoci_tardivi, "precoci_tardivi"))
        t2_SVM.start()
        t1_SVM.start()
        t2_SVM.join()
        t1_SVM.join()


    m, s = divmod(time() - start, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02f" % (h, m, s))
    print("end of program")

if __name__ == "__main__":
    main()
