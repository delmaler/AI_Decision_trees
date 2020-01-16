import sklearn
import numpy as np
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from operator import itemgetter, attrgetter


def normalize(sample, max_arr, min_arr):
    normalized = []
    k = 0
    for val in sample:
        normalized.append((val-min_arr[k])/(max_arr[k]-min_arr[k]))
        k += 1
    return normalized


def get_distance(sample_a, sample_b):
    j = 0
    sum = 0
    for val in sample_a:
        sum += (val-sample_b[j])**2
        j += 1
    return np.sqrt(sum)


def get_decision(data_set, outcomes, sample):
    distances = []
    index = 0
    desicion = 0
    for line in data_set:
        distances.append((get_distance(sample, line), index))
        index += 1
    new_sorted = sorted(distances, key=lambda tup: tup[0])
    index = 0
    while index < 9:
        temp = outcomes[new_sorted[index][1]]
        val = 4 if temp == 1 else -1
        desicion += val
        index += 1
    return 1 if desicion > 0 else 0


if __name__ == '__main__':
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
             'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age']

    #setting train data
    train_data = pd.read_csv("train.csv", header=None, names=columns, skiprows=1)
    samples_train = train_data[features]
    results_train = train_data.Outcome
    np_train_data = np.array(samples_train)
    np_train_result = np.array(results_train)
    max_vals_train = np_train_data.max(0)
    min_vals_train = np_train_data.min(0)

    i = 0
    for row in np_train_data:
        np_train_data[i] = normalize(row, max_vals_train, min_vals_train)
        i += 1
    #setting tests data
    test_data = pd.read_csv("test.csv", header=None, names=columns, skiprows=1)
    samples_test = test_data[features]
    results_test = test_data.Outcome
    np_test_data = np.array(samples_test)
    np_test_result = np.array(results_test)
    max_vals_test = np_test_data.max(0)
    min_vals_test = np_test_data.min(0)

    i = 0
    for row in np_test_data:
        np_test_data[i] = normalize(row, max_vals_train, min_vals_train)
        i += 1

    #setting results
    results = []
    for row in np_test_data:
        results.append(get_decision(np_train_data, np_train_result, row))
    print (confusion_matrix( np_test_result,results))

