from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import pandas as pd


def print_confusion_mat(outcome, prediction):
    confusion_mat = metrics.confusion_matrix(outcome, prediction)
    TN, TP, FN, FP = confusion_mat.ravel()
    print("[[{} {}]".format(TP, FP))
    print("[{} {}]]".format(FN, TN))
    """
    error_w=4*FN+FP
    print("the error is %d",error_w)
    return error_w
    """

train_df = pd.read_csv("train.csv", sep=",")
cols = list(train_df.columns.values)
features = cols[:-1]
#checking how many positive examples there are
num_positives=train_df['Outcome'].sum()
li=[]
li.append(train_df.sort_values('Outcome')[:num_positives])
li.append(train_df.sort_values('Outcome')[-num_positives:])
#creating the new balanced datafile
modified_df=pd.concat(li,axis=0,ignore_index=True)
decision_tree = DecisionTreeClassifier(
    "entropy").fit(modified_df[features], modified_df.Outcome)
test_df = pd.read_csv("test.csv", sep=",")
prediction = decision_tree.predict(test_df[features])
print_confusion_mat(test_df.Outcome, prediction)
#code to find best value for error_w
"""
error_w=300
index=0
for x in range (2,100):
    decision_tree = DecisionTreeClassifier(
        "entropy",min_samples_split=x).fit(modified_df[features], modified_df.Outcome)
    test_df = pd.read_csv("test.csv", sep=",")
    prediction = decision_tree.predict(test_df[features])
    cut_error_w=print_confusion_mat(test_df.Outcome, prediction)
    if error_w > cut_error_w:
        error_w=cut_error_w
        index=x
print(error_w,index)
"""