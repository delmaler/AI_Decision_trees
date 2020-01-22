from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pandas as pd
import numpy as np


def print_confusion_mat(outcome, prediction):
    confusion_mat = metrics.confusion_matrix(outcome, prediction)
    TN, TP, FN, FP = confusion_mat.ravel()
    print("[[{} {}]".format(TP, FP))
    print("[{} {}]]".format(FN, TN))
    # error_w=4*FN+FP
    #print("the error is %d",error_w)


train_df = pd.read_csv("train.csv", sep=",")
test_df = pd.read_csv("test.csv", sep=',')
cols = list(train_df.columns.values)
features = cols[:-1]
Dtree = DecisionTreeClassifier(criterion='entropy', min_samples_split=9,
                               class_weight={0: 1, 1: 4})

Dtree = Dtree.fit(train_df[features], train_df.Outcome)
prediction = Dtree.predict(test_df[features])
print_confusion_mat(test_df.Outcome, prediction)
