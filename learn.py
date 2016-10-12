import csv
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
def read_files():
    global hero, test, train, sample
    with open('Data/MatchDetail.csv', 'r') as csvfile:
        rows = csv.reader(csvfile)
        hero = [row for row in rows]
    with open('Data/MatchOverviewTest.csv', 'r') as csvfile:
        rows = csv.reader(csvfile)
        test = [row for row in rows]
    with open('Data/MatchOverviewTraining.csv', 'r') as csvfile:
        rows = csv.reader(csvfile)
        train = [row for row in rows]
    with open('Data/sampleSubmission.csv', 'r') as csvfile:
        rows = csv.reader(csvfile)
        sample = [row for row in rows]
    def convert_to_int(array):
        for i in range(1, len(array)):
            for j in range(len(array[0])):
                try:
                    array[i][j] = int(array[i][j])
                except ValueError:
                    array[i][j] == int(bool(array[i][j]))
    for lst in [hero, test, train, sample]:
        convert_to_int(lst)


read_files()

X = [0] * 228
for i in range(1,227):
    X.append([(i in ).astype('int') for i in range(1, 227)])
"""Set the second five's columns to 113+themselves
   Make a new DataFrame saying whether or not the hero is in the game, using the
   hero id as the column titles.
   Add two more columns to this DataFrame computing the difference of the
   average of the gold & xp for each team at the end of the game.
[1 for i in range(10) if train[i+1]]
model = LogisticRegression()
model = model.fit(x, y)
model.score(x, y)
