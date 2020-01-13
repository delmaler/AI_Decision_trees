import sklearn
import numpy as np
import csv
import binarytree
import pandas as pd
class DataEntry:
    def __init__(
        self,
        Pregnancies: int,
        Glucose:int,
        BloodPressure:int,
        SkinThickness:int,
        Insulin:int,
        BMI:float,
        DiabetesPedigreeFunction:float,
        Age:int,
        Outcome:bool


    ):
        self.Pregnancies =Pregnancies
        self.Glucose = (Glucose)
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age= Age
        self.Outcome =Outcome
        
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
""" TODO load data into features and data set"""


if __name__ == '__main__':
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
             'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age']
    train_data = pd.read_csv("train.csv", header=None, names=columns, skiprows=1)
    X_train = train_data[features]
    y_train = train_data.Outcome



