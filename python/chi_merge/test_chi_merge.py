#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_selection import SelectKBest, chi2

def chimerge(data, attr, label, max_intervals):
    distinct_vals = sorted(set(data[attr])) # Sort the distinct values
    labels = sorted(set(data[label])) # Get all possible labels
    empty_count = {l: 0 for l in labels} # A helper function for padding the Counter()
    intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))] # Initialize the intervals for each attribute
    while len(intervals) > max_intervals: # While loop
        chi = []
        for i in range(len(intervals)-1):
            # Calculate the Chi2 value
            obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
            obs1 = data[data[attr].between(intervals[i+1][0], intervals[i+1][1])]
            total = len(obs0) + len(obs1)
            count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0[label])}.items()])
            count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1[label])}.items()])
            count_total = count_0 + count_1
            expected_0 = count_total*sum(count_0)/total
            expected_1 = count_total*sum(count_1)/total
            chi_ = (count_0 - expected_0)**2/expected_0 + (count_1 - expected_1)**2/expected_1
            chi_ = np.nan_to_num(chi_) # Deal with the zero counts
            chi.append(sum(chi_)) # Finally do the summation for Chi2
        min_chi = min(chi) # Find the minimal Chi2 for current iteration
        for i, v in enumerate(chi):
            if v == min_chi:
                min_chi_index = i # Find the index of the interval to be merged
                break
        new_intervals = [] # Prepare for the merged new data array
        skip = False
        done = False
        for i in range(len(intervals)):
            if skip:
                skip = False
                continue
            if i == min_chi_index and not done: # Merge the intervals
                t = intervals[i] + intervals[i+1]
                new_intervals.append([min(t), max(t)])
                skip = True
                done = True
            else:
                new_intervals.append(intervals[i])
        intervals = new_intervals
    for i in intervals:
        print('[', i[0], ',', i[1], ']', sep='')


def main():
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder()

    # 调用函数参数示例
    iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    columns = iris.columns.values
    #enc.fit(iris.loc[:, [4]].values.reshape(-1, 1))
    enc.fit(iris.loc[:, [4]].values)
    #print(enc.transform(iris.loc[:, 4].values.reshape(-1, 1)))
    iris.loc[:, 4] = enc.transform(iris.loc[:, 4].values.reshape(-1, 1)).reshape(-1,)

    #iris.loc[iris[4] == "Iris-virginica", 4] = 1
    #iris.loc[iris[4] != 1, 4] = 0
    print("++++++++++++++++++++++++++++++")
    for column in columns[:-1]:
        print("========================================")
        chimerge(iris, column, 4, max_intervals=6)
        #print("Column: {}\n{}".format(column, bins))
    print("Chi2:\n")
    print(chi2(iris[[0,1,2,3]].values, iris[4].values))


if __name__ == "__main__":
    main()
