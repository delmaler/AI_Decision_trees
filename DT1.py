import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import csv
import binarytree
import pandas as pd

"""       
def Enthropy(data_set):
    enthropy=0
    EC=[0,0]
    for data_entry in data_set:
        EC[data_entry.outcome]+=1
    PC=[EC[0]/len(data_set),EC[1]/len(data_set)]
    for i in range(2):
        if PC[i]>0:
            enthropy-=PC[i]*np.log2(PC[i])
    return enthropy

def InformationGain(data_set,feature):
    enthropy=Enthropy(data_set)
    children_enthropy=0
    data_set_size=len(data_set)
    for value in feature:
        child_data_set=filter(lambda x: x.feature==value , data_set) #TODO need to support continious features
        children_enthropy+=Enthropy(child_data_set)*(len(child_data_set)/data_set_size)
    return enthropy-children_enthropy

def ID3_SelectFeature(features: list ,data_set,x:int):
    max_information_gain=0
    max_feature=DataEntry.Pregnancies
    for feature in features:
        information_gain=InformationGain(data_set,feature)
        if InformationGain(data_set,feature)>max_information_gain:
            max_feature=feature
            max_information_gain=information_gain
    return max_feature
    """
""" TODO load data into features and data set"""

criteria = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age']
train_data = pd.read_csv("train.csv", sep="," , names=criteria, skiprows=1)
print(train_data.head())
decision_tree = DecisionTreeClassifier(
    "entropy").fit(train_data[features], train_data.Outcome)
data = pd.read_csv("test.csv", sep=",",header=None, names=criteria, skiprows=1)
classificator = decision_tree.predict(data[features])
confusion_mat = metrics.confusion_matrix(data.Outcome, classificator)
print(confusion_mat)
TN, TP, FN, FP = confusion_mat.ravel()
print("[[{} {}]".format(TP, FP))
print("[{} {}]]".format(FN, TN))
pruning_values = [3,9,27]
accuracy_values = []
for x in pruning_values:
    acc=0
    for i in range(100):
        dt = DecisionTreeClassifier(criterion="entropy", min_samples_split=x)
        dt = dt.fit(train_data[features], train_data.Outcome)
        prediction = dt.predict(data[features])
        acc += metrics.accuracy_score(data.Outcome,prediction)
    accuracy_values.append(acc/100)
print(accuracy_values)

