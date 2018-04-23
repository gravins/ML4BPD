import pydotplus as pydotplus
import itertools
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

def visualize_tree(tree, feature_names, name=None):
    """
    Create a png of a tree plot using graphviz
    :param tree: scikit-learn DecsisionTree
    :param feature_names: list of feature names
    :param name: name of file .png
    :param n_model: number of model used to predict (only for classification with one model per output)
    """

    f = export_graphviz(tree, out_file=None,
                        feature_names=feature_names,
                        filled=True,
                        rounded=True)

    graph = pydotplus.graph_from_dot_data(f)

    if name is not None:
        #graph.write_dot(name+".dot")
        graph.write_png(name+".png")
    else:
        #graph.write_dot("decisionTreePlot.dot")
        graph.write_png("decisionTreePlot.png")


def plot_nested_test_score(nested, stdnested, auc, label):
    """
    Create plot of nested score and test score
    :param nested: List of nested score value
    :param stdnested: List of nested score value's std
    :param auc: List of nested score value
    :param label: List of labels name
    """
    pos = list(range(len(label)))

    # create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(pos, auc, color="black", marker="x", markersize=4, linestyle=" ", label="test set score")
    ax.errorbar(pos, nested, fmt="o", markersize=3, color="black", label="nested CV score",
                yerr=stdnested, ecolor='black', capsize=3)
    ax.set_ylabel('ROC AUC score')
    ax.set_xticks(pos)
    ax.set_xticklabels(label)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
    plt.legend(loc="best")

    plt.tight_layout()
    # save plot to file
    plt.savefig('result_nested_test.png')


def plot_nested_validation_score(val, std ,label):
    """
    Create plot of nested score and validation score
    :param val: list of value to plot
        val[0]: contains list of nested score
        val[1]: contains list of validation score
    :param std: list of stds value to plot
        std[0]: contains list of nested std
        std[1]: contains list of validation std
    :param label: list of label
    """

    pos = list(range(len(label)))
    width = 0.1
    color = ['black', 'grey']
    mark = ["o", "x"]
    legend = ["nested CV score", "CV score"]
    # create plot
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in [0, 1]:
        ax.errorbar([p + width * i for p in pos], val[i], fmt=mark[i], markersize=4, color=color[i], label=legend[i],
                    yerr=std[i], ecolor=color[i], capsize=3)
    ax.set_ylabel('ROC AUC score')

    # Set the position of the x ticks
    ax.set_xticks([p + (width / 2) for p in pos])

    ax.set_xticklabels(label)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
    plt.legend(loc="best")

    plt.tight_layout()
    # save plot to file
    plt.savefig('result_nested_and_cv.png')
    plt.close(fig)


def histogram(value, xlabel, name=None, ylabel=None, title=None ):
    """
    Print histogram with value attach to x-axis's labels
    :param value: list of value
    :param xlabel: x-axis's labels
    :param name: file's name
    :param ylabel: y-axis's label
    :param title: histogram's title
    """

    # Setting the positions and width for the bars and color
    pos = list(range(len(xlabel)))
    width = 0.25
    color = 'red'
    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar in position pos + some width buffer,
    rects = plt.bar([p + width for p in pos],
            #using value data,
            value,
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color=color)

    # Set the y axis label
    if ylabel is None:
        ax.set_ylabel('features importance')
    else:
        ax.set_ylabel(ylabel)

    if not title is None:
        # Set the chart's title
        ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p + 1 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(xlabel)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*4)

    ax.axhline(0, color='black')

    plt.tight_layout()

    # Adding the legend and showing the plot

    plt.grid(ls='dotted')

    if name is None:
        fig.savefig("histogram_" + title + ".png")
    else:
        fig.savefig(name + ".png")

    plt.close(fig)


def randomForestPlot(n_estimators, X, y, X_tr, y_tr, top_param=None, name=None, ylabel=None, title=None):
    """
    Plot Random Forest performance
    :param top_param: best params for random forest
    :param n_estimators: max number of tree in the forest
    :param X: input of test set
    :param y: output of test set
    :param X_tr: input of training set
    :param y_tr: output of training set
    :param name: file's name
    :param ylabel: y-axis's label
    :param title: plot's title
    """

    score = [0] * (n_estimators)
    scoreTR = [0] * (n_estimators)
    oobScore = [0] * (n_estimators)
    if top_param is not None:
        score_topparam = [0] * (n_estimators)
        scoreTR_topparam = [0] * (n_estimators)
        oobScore_topparam = [0] * (n_estimators)

    # Using the new warm_start parameter, you could add trees 1 by 1
    rf = RandomForestClassifier(n_jobs=1, warm_start=True, oob_score=True)

    if top_param is not None:
        top_param["oob_score"] = True
        rf_topparam = RandomForestClassifier(n_jobs=1, warm_start=True, **top_param)

    for i in range(1, n_estimators + 1):
        rf.set_params(n_estimators=i)
        rf.fit(X_tr, y_tr)
        score[i-1] = f1_score(y, rf.predict(X), average="micro")
        scoreTR[i - 1] = f1_score(y_tr, rf.predict(X_tr), average="micro")
        oobScore[i-1] = 1 - rf.oob_score_
        if top_param is not None:
            rf_topparam.set_params(n_estimators=i)
            rf_topparam.fit(X_tr, y_tr)
            score_topparam[i - 1] = f1_score(y, rf_topparam.predict(X), average="micro")
            scoreTR_topparam[i - 1] = f1_score(y_tr, rf_topparam.predict(X_tr), average="micro")
            oobScore_topparam[i - 1] = 1 - rf_topparam.oob_score_



    # Draw plot for F1 score
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(np.arange(n_estimators) + 1, score,
            label='RF Test',
            color='red')
    ax.plot(np.arange(n_estimators) + 1, scoreTR,
            label='RF Training',
            color='#0091EA')
    if top_param is not None:
        ax.plot(np.arange(n_estimators) + 1, score_topparam,
                label='RF top params Test',
                color='yellow')
        ax.plot(np.arange(n_estimators) + 1, scoreTR_topparam,
                label='RF top params Training',
                color='#00E676')

    ax.set_xlim(0, n_estimators + 10)
    ax.set_xlabel("n_estimators")
    plt.legend(loc="best")

    # Set the y axis label
    if ylabel is None:
        ax.set_ylabel("F1 score")
    else:
        ax.set_ylabel(ylabel)

    if not title is None:
        # Set the chart's title
        ax.set_title(title)

    if name is None:
        txt = "randomforest_f1_score"
        if not title is None:
            fig.savefig(txt + "_" + title + ".png")
        else:
            fig.savefig(txt+".png")
    else:
        fig.savefig(name + ".png")

    plt.close(fig)

    # Draw plot for OOB error rate
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(np.arange(n_estimators) + 1, oobScore,
            label='Random Forest',
            color='#0091EA')
    if top_param is not None:
        ax1.plot(np.arange(n_estimators) + 1, oobScore_topparam,
                label='Random Forest top params',
                color='#00E676')

    ax1.set_xlim(0, n_estimators + 10)
    ax1.set_xlabel("n_estimators")
    ax1.set_ylabel("OOB error rate")

    if not title is None:
        # Set the chart's title
        ax1.set_title(title)

    plt.legend(loc="best")
    fig1.savefig("randomforest_OOB_score_"+title+".png")

    plt.close(fig1)

    del rf
    if top_param is not None:
        del rf_topparam


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix
    :param cm: confusion matrix
    :param classes: classes used to create confusion matrix
    :param normalize: if true apply normalization
    :param title: title of png
    :param cmap:
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(9, 9))
    res = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(res)

    tick_marks = np.arange(len(classes))
    ax.set_xticklabels(classes)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.set_yticklabels(classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #plot.text(x, y, string, fontdict=None, withdash=False, **kwargs)
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    plt.tight_layout()
    if title is not None:
        plt.savefig(title+".png")
    else:
        plt.savefig("Confusion matrix_"+ ".png")
