import pandas as pd


def getDataFramefromExcel(path):
    """
    Function that make DataFrame from excel file

    :param path: str , path of the file to read
    :return: DataFrame from excel file
    """
    database = pd.read_csv(path)

    return database


def saveDataFrameintoExcel(df, path):
    """
    Function that save DataFrame into excel file

    :param df: DataFrame
    :param path: str
    """

    # Write DataFrame to a excel sheet
    df.to_csv(path)



def splitFrom(df, attr, val):
    """
    Split DataFrame in two subset based on year attribute
    :param df: DataFrame to split
    :param attr: attribute on which split data
    :param val: value of attribute where do split
    :return: two subset
    """
    if attr not in df.columns:
        raise ValueError("******* "+attr+" not in DataFrame *******")
    subfd1 = df.loc[df[attr] < val]
    subfd2 = df.loc[df[attr] >= val]
    return subfd1, subfd2


def orderColumns(df):
    """
    Function that order DataFrame's columns according to other, outcome, possible early risk factors, possible late risk factors
    :param df: DataFrame
    :return: ordered DataFrame
    """
    col = df.columns.tolist()

    early = ["drsurf", "zbw", "sga10", "race"]
    for i in range(col.index("bwgt"), col.index("drcpap")+1):
        early.append(col[i])

    outcome = ["newox28", "bpd", "anyvent36w"]
    for i in range(col.index("ox36"), col.index("sterbpd") + 1):
        outcome.append(col[i])
    late = ["rds", "pntx"]
    for i in range(col.index("anyvent"), col.index("pdatrattato") + 1):
        late.append(col[i])

    col = [x for x in col if x not in outcome and x not in late and x not in early] # now col contains all elements in "other"
    col = col + outcome
    col = col + early
    col = col + late

    return df[col]
