from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
#from sklearn.tree import export_graphviz
import pandas as pd
#import matplotlib.pyplot as plt
# from graphviz import render


"""       
def Enthropy(test_df_set):
    enthropy=0
    EC=[0,0]
    for test_df_entry in test_df_set:
        EC[test_df_entry.outcome]+=1
    PC=[EC[0]/len(test_df_set),EC[1]/len(test_df_set)]
    for i in range(2):
        if PC[i]>0:
            enthropy-=PC[i]*np.log2(PC[i])
    return enthropy

def InformationGain(test_df_set,feature):
    enthropy=Enthropy(test_df_set)
    children_enthropy=0
    test_df_set_size=len(test_df_set)
    for value in feature:
        child_test_df_set=filter(lambda x: x.feature==value , test_df_set) #TODO need to support continious features
        children_enthropy+=Enthropy(child_test_df_set)*(len(child_test_df_set)/test_df_set_size)
    return enthropy-children_enthropy

def ID3_SelectFeature(features: list ,test_df_set,x:int):
    max_information_gain=0
    max_feature=test_dfEntry.Pregnancies
    for feature in features:
        information_gain=InformationGain(test_df_set,feature)
        if InformationGain(test_df_set,feature)>max_information_gain:
            max_feature=feature
            max_information_gain=information_gain
    return max_feature
    """


def print_confusion_mat(outcome, prediction):
    confusion_mat = metrics.confusion_matrix(outcome, prediction)
    TN, TP, FN, FP = confusion_mat.ravel()
    print("[[{} {}]".format(TP, FP))
    print("[{} {}]]".format(FN, TN))
    #error_w=4*FN+FP
    #print("the error is %d",error_w)

train_df = pd.read_csv("train.csv", sep=",")
cols = list(train_df.columns.values)
features = cols[:-1]
decision_tree = DecisionTreeClassifier(
    "entropy").fit(train_df[features], train_df.Outcome)
test_df = pd.read_csv("test.csv", sep=",")
prediction = decision_tree.predict(test_df[features])
print_confusion_mat(test_df.Outcome, prediction)


"""
dt = DecisionTreeClassifier(criterion="entropy", min_samples_split=27).fit(train_df[features],train_df['Outcome'])
export_graphviz(dt, out_file='tree.dot', filled=True, rounded=True, special_characters=True,
                feature_names=features, class_names=['0', '1'])
render('dot', 'png', 'tree.dot')
"""

#code for making graph
"""
pruning_values = range(2,27)
accuracy_values = []
for x in pruning_values:
    acc=0
    for i in range(20):
        dt = DecisionTreeClassifier(criterion="entropy", min_samples_split=x)
        dt = dt.fit(train_df[features], train_df.Outcome)
        prediction = dt.predict(test_df[features])
        acc += metrics.accuracy_score(test_df.Outcome,prediction)
    accuracy_values.append(acc/20)
print(accuracy_values)
print(max(accuracy_values),accuracy_values.index(max(accuracy_values)))
"""


