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
    hero = pd.read_csv('Data/MatchDetail.csv')
    test = pd.read_csv('Data/MatchOverviewTest.csv')
    train = pd.read_csv('Data/MatchOverviewTraining.csv')
    sample = pd.read_csv('Data/sampleSubmission.csv')


read_files()

"""Set the second five's columns to 113+themselves
   Make a new DataFrame saying whether or not the hero is in the game, using the
   hero id as the column titles.
   Add two more columns to this DataFrame computing the difference of the
   average of the gold & xp for each team at the end of the game.
[1 for i in range(10) if train[i+1]]
model = LogisticRegression()
model = model.fit(x, y)
model.score(x, y)"""
