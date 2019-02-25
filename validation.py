from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, cross_validate
import itertools
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import f1_score, roc_auc_score
import json
import threading

from dask.diagnostics import ProgressBar
import dask_searchcv as dcv

def fit_and_result(estimator, x_tr, y_tr, x_val, y_val, params, result, f=None):
    """
    FUnction used to test one setting
    :param estimator: scikit-learn estimator
    :param x_tr: input of training set
    :param y_tr: output of training set
    :param x_val: input of validation set
    :param y_val: output of validation set
    :param params: params setting
    :param result: global result's list
    :param f: name of file on which save result
    """
    if (("bootstrap" in params and "oob_score" in params) and (not(params["bootstrap"]==False and params["oob_score"]==True)))\
            or (("bootstrap" not in params or "oob_score" not in params)):

        estimator.set_params(**params)
        estimator.fit(x_tr, y_tr)
        pred_val = estimator.predict(x_val)
        pred_tr = estimator.predict(x_tr)
        partial_res = {"params": params,
                       "AUC_val": roc_auc_score(y_val, pred_val),
                       "AUC_train": roc_auc_score(y_tr, pred_tr),
                       "f1_val": f1_score(y_val, pred_val, average="micro"),
                       "f1_train": f1_score(y_tr, pred_tr, average="micro")}

        result.append(partial_res)
        if f is not None:
            lock.acquire()
            try:
                json.dump(partial_res, f)
            finally:
                lock.release()


lock = threading.Lock()


def gridsearchValidation(estimator, param_grid, x_tr, y_tr, x_val, y_val, file):
    """
    Function that do simplest validation
    :param estimator: scikit-learn estimator used for validation
    :param param_grid: grid of params on which perform validation
    :param x_tr: input of training set
    :param y_tr: output of training set
    :param x_val: input of validation set
    :param y_val: output of validation set
    :param file: name of file on which save result
    :return: dictionary with best param obtained with ROC-AUC score and F1 score
    """

    executor = ThreadPoolExecutor(max_workers=20)
    result = []
    f = open("val_results_"+file+".json", "a")
    labels, terms = zip(*param_grid.items())
    i = 0
    for params in [dict(zip(labels, term)) for term in itertools.product(*terms)]:
        i += 1
        if i == 1 or i % 50 == 0:
            print(file, ":", str(i), " ", params)
        executor.submit(fit_and_result(estimator, x_tr, y_tr, x_val, y_val, params, result, f))

    # shutdown the threadpool and wait end of all worker
    executor.shutdown(wait=True)

    bestAUC = result[0]
    for i in range(1, len(result)):
        if result[i]["AUC_val"] > bestAUC["AUC_val"]:
            bestAUC = result[i]
    bestF1 = result[0]
    for i in range(1, len(result)):
        if result[i]["f1_val"] > bestF1["f1_val"]:
            bestF1 = result[i]
    print("AUC best ", bestAUC)
    print("F1 best ", bestF1)
    return {"auc": bestAUC["params"], "f1": bestF1["params"]}


def nestedCrossValidation(estimator, grid, x, y, inner_split, outer_split, file_name="", shuffle=True, n_jobs=1, n_iter_search = None):
    """
    Function that perform  nestesd cross validation.
    :param estimator: scikit-learn estimator
    :param grid: [dict] parameter settings to test
    :param x: features
    :param y: targets
    :param inner_split: number of split into inner cross validation
    :param outer_split: number of split into inner cross validation
    :param shuffle: if True shuffle data before splitting
    """
    grid_search = 0
    for el in grid:
        if type(grid[el]) is list:
            grid_search += 1
    if grid_search == len(grid):
        grid_search = True
    else:
        grid_search = False

    if grid_search == False and n_iter_search is None:
        ValueError("You must specify the number of iteration for random search")

    inner_cv = KFold(n_splits=inner_split, shuffle=shuffle)
    outer_cv = KFold(n_splits=outer_split, shuffle=shuffle)

    cls = None
    if grid_search == True:
        clf = GridSearchCV(estimator=estimator, param_grid=grid, cv=inner_cv, scoring="roc_auc", n_jobs=n_jobs)
    else:
        clf = RandomizedSearchCV(estimator=estimator, param_distributions=grid, n_iter=n_iter_search, cv= inner_cv, scoring="roc_auc", n_jobs=n_jobs, random_state=42)

    nested_score = cross_val_score(clf, scoring="roc_auc", X=x, y=y, cv=outer_cv, n_jobs=n_jobs)
    #nested_score = cross_validate(clf, scoring="roc_auc", X=x, y=y, cv=outer_cv, n_jobs=n_jobs)
    #print(nested_score)
    return {"nested_score": nested_score, "mean": nested_score.mean(), "std": nested_score.std()}


def report(grid_scores, n_top=3):
    """
    Report top n_top parameters settings, default n_top=3

    :param grid_scores: output from grid or random search
    :param n_top: how many to report, of top models
    :return: top_params: [dict] top parameter settings found in
                  search
    """
    if len(grid_scores["mean_test_score"]) < n_top:
        n_top = len(grid_scores["mean_test_score"])

    top_scores = {"params": [], "mean_test_score": [], "std_test_score": []}
    if "mean_train_score" in grid_scores.keys():
        top_scores["mean_train_score"] = []
        top_scores["std_train_score"] = []

    rank = grid_scores["rank_test_score"].tolist()
    i = 1
    while n_top > 0:
        ii = [ind for ind, val in enumerate(rank) if val == i]
        if n_top < len(ii):
            for k in top_scores.keys():
                for ind in ii[:n_top]:
                    top_scores[k].append(grid_scores[k][ind])
            n_top = 0
        else:
            for k in top_scores.keys():
                for ind in ii:
                    top_scores[k].append(grid_scores[k][ind])
            n_top = n_top - len(ii)
        i = (i + len(ii))

    return top_scores


def run_gridsearch(X, y, clf, param_grid, cv=5, n_jobs=3):
    """
    Run a grid search for best estimator parameters.
    :param X: features
    :param y: targets (classes)
    :param clf: scikit-learn classifier
    :param param_grid: [dict] parameter settings to test
    :param cv: fold of cross-validation, default 5
    :return top_params: [dict] from report()
    """
    grid_search = dcv.GridSearchCV(clf,
                        scoring="roc_auc",
                        param_grid=param_grid,
                        cv=cv, n_jobs=n_jobs)
    with ProgressBar():
            gs = grid_search.fit(X, y)

    top_params = report(grid_search.cv_results_, 5)

    return top_params


def run_randomsearch(X, y, clf, param_dist, cv=5, n_iter_search=20, n_jobs=3):
    """
    Run a random search for best Decision Tree parameters.
    :param X: features
    :param y: targets (classes)
    :param cf: scikit-learn classifier
    :param param_dist: [dict] list, distributions of parameters to sample
    :param cv: fold of cross-validation, default 5
    :param n_iter_search: number of random parameter sets to try, default 20.
    :return top_params: [dict] from report()
    """
    random_search = dcv.RandomizedSearchCV(clf,
                        scoring="roc_auc",
                        param_distributions=param_dist,
                        n_iter=n_iter_search,
                        cv=cv, n_jobs=n_jobs,
                        random_state=42)
    with ProgressBar():
        rs = random_search.fit(X, y)

    top_params = report(random_search.cv_results_, 5)

    return top_params
