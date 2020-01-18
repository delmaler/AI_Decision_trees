import sklearn
import numpy as np
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from operator import itemgetter, attrgetter
from itertools import combinations
from sklearn.metrics import accuracy_score


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def convert_words_to_index(partial, full):
    res = []
    for l in partial:
        res.append(full.index(l))
    return res


def normalize(sample, max_arr, min_arr):
    normalized = []
    k = 0
    for val in sample:
        normalized.append((val-min_arr[k])/(max_arr[k]-min_arr[k]))
        k += 1
    return normalized


def normalized_whole_data_set(data_set, mim_max_from):
    res = []
    max_vals_train = mim_max_from.max(0)
    min_vals_train = mim_max_from.min(0)
    for row in data_set:
        res.append(normalize(row, max_vals_train, min_vals_train))
    return res


def get_combinations(input):
    res = []
    for i in range(0, len(input)+1):
        temp_combi = [list(j) for j in combinations(input,i)]
        if len(temp_combi) > 0:
            res.extend(temp_combi)
    return res


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
        val = 1 if temp == 1 else -1
        desicion += val
        index += 1
    return 1 if desicion > 0 else 0


def get_precision_for_feature_group(train_data_set, train_results, test_data_set, test_result):
    results = []
    for row in test_data_set:
        results.append(get_decision(train_data_set, train_results, row))
    return accuracy_score(test_result, results)


def get_best_feature_group(samples_train, results_train, samples_test, results_test, features):
    max = 0
    remaining = features
    result = []
    chosen = []
    flag = True
    while flag:
        d = 0
        flag = False
        to_remove = []
        chosen = result
        while d < len(remaining):
            feature = remaining[d]
            chosen.append(feature)
            temp_samples_train = samples_train[chosen]
            temp_samples_test = samples_test[chosen]
            np_samples_train = np.array(temp_samples_train)
            np_samples_test = np.array(temp_samples_test)
            min_max_from = np_samples_train
            np_samples_train = normalized_whole_data_set(np_samples_train, min_max_from)
            np_samples_test = normalized_whole_data_set(np_samples_test, min_max_from)
            temp = get_precision_for_feature_group(np_samples_train, results_train, np_samples_test, results_test)
            if temp > max:
                max = temp
                to_remove = feature
                flag = True
            d += 1
            chosen.pop()
        if flag:
            result.append(to_remove)
            remaining.remove(to_remove)
    return result


if __name__ == '__main__':
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
               'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age']
    temp_feature = features.copy()
    #setting train data
    train_data = pd.read_csv("train.csv", header=None, names=columns, skiprows=1)
    samples_train = train_data[features]
    results_train = train_data.Outcome
    test_data = pd.read_csv("test.csv", header=None, names=columns, skiprows=1)
    samples_test = test_data[features]
    results_test = test_data.Outcome
    res_in_words = get_best_feature_group(samples_train, results_train, samples_test, results_test, temp_feature)
    print(convert_words_to_index(res_in_words, features))
    #setting results
