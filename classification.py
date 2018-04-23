from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import validation
from sklearn.base import clone
import pickle


def classification(mod, param_grid, x_tr, y_tr, x_ts, y_ts, file_name, valid=None, n_job=3):
    """
    :param mod: scikit-learn estimator used for classification
    :param param_grid: dict of params for grid search or random search
                        the dict for grid search contains list and that for
                        random search contains randint
    :param x_tr: input for training set
    :param y_tr: array of list of target
    :param x_ts: input for test set
    :param y_ts: array of list of target
    :param file_name: name of file in which save result
    :param valid: if value is int mean percentage of data for simple validation, if is bool mean cross validation
    :param n_job: number of job to run in parallel
    """
    if not ("int" in str(type(valid)) or "bool" in str(type(valid)) or valid is None):
        raise ValueError("******* valid can be bool, int or None *******")

    mod_gs = None
    if valid is not None:

        # Check if we are doing grid or random search
        grid_search = 0
        for el in param_grid:
            if type(param_grid[el]) is list:
                grid_search += 1
        if grid_search == len(param_grid):
            grid_search = True
        else:
            grid_search = False

        # Clone the model
        mod_gs = clone(mod)
        x_tr_split, x_val, y_tr_split, y_val = None, None, None, None
        if "int" in str(type(valid)) and grid_search == False:
            # if we are doing simple validation
            # split data into tr_set and validation_set
            x_tr_split, x_val, y_tr_split, y_val = train_test_split(x_tr, y_tr, test_size=valid)
            ts_gs = validation.gridsearchValidation(mod, param_grid, x_tr_split, y_tr_split, x_val, y_val, file_name)
            mod_gs.set_params(**ts_gs["auc"])

        if "bool" in str(type(valid)):
            if grid_search == True:
                ts_gs = validation.run_gridsearch(X=x_tr, y=y_tr, clf=mod, param_grid=param_grid, n_jobs=n_job)
            else:
                ts_gs = validation.run_randomsearch(X=x_tr, y=y_tr, clf=mod, param_dist=param_grid, n_iter_search=500, n_jobs=n_job)
            mod_gs.set_params(**ts_gs["params"][0])

        # Save best params after validation into pickle file
        used_estimator = str(type(mod)).split(".")
        used_estimator = used_estimator.pop()[:-2]
        pickle.dump(ts_gs, open(used_estimator+"_"+file_name+".p", "wb"))

        mod_gs.fit(x_tr, y_tr)

        # Run nested CV
        nested_result = None
        if grid_search == True:
            # nested cv with grid search
            nested_result = validation.nestedCrossValidation(estimator=mod, grid=param_grid, x=x_tr, y=y_tr, inner_split=5,
                                                         outer_split=3, n_jobs=n_job, file_name=file_name)
        else:
            # nested cv with random search
            nested_result = validation.nestedCrossValidation(estimator=mod, grid=param_grid, x=x_tr, y=y_tr,
                                                             inner_split=5,
                                                             outer_split=3, n_jobs=n_job, file_name=file_name, n_iter_search=20)

        if isinstance(mod, RandomForestClassifier) or isinstance(mod, DecisionTreeClassifier):
            result_gs = {"best_params": ts_gs,
                         "features_imp": str(list(zip(x_tr.columns, mod_gs.feature_importances_))),
                         "nested": nested_result}
        else:
            result_gs = {"best_params": ts_gs,
                         "nested": nested_result}
        pickle.dump(result_gs, open(used_estimator + "_RESULT_" + file_name + ".p", "wb"))



    if valid is not None:
        return {"norm": mod, "valid": result_gs}
    else:
        return {"norm": mod}


def scoring(res, y_true, l):
    pos = 0
    i = 0

    for y in y_true:
        if res[i] == y:
            pos += 1
        i += 1
    cm = confusion_matrix(y_true, res)
    #printerTree.plot_confusion_matrix(cm,[0,1])
    print(l, " confusion matrix: \n", cm)
    print(l, " correct result: " + str(pos) + "/" + str(len(res)))
    print(l, " Classification f1_score is ", f1_score(y_true, res, average="micro"))
    print(l, " Classification roc_auc_score is ", roc_auc_score(y_true, res))
