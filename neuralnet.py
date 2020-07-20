# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:47:52 2020

@author: anujr
"""


import tensorflow
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('dataset.csv')
y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(dataset[['l', 'b', 'h']], y, test_size=0.2)
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = Sequential()
model.add(layers.Dense(3, activation='relu'))
#model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
t = model.fit(x=X_train, y=y_train, validation_data = (X_test, y_test), epochs = 100)
X_test = pd.read_csv('test_data.csv')[['l','b','h']].to_numpy()
y_test = pd.read_csv('test_data.csv')[['label']].to_numpy()
X_test = sc.fit_transform(X_test)
pred = model.predict(X_test)
ans = []
for val in pred:
    if val < .5:
        ans.append(0)
    else:
        ans.append(1)
failure = 0
total = 0
for i, j in zip(ans, y_test):
    total += 1
    if i != int(j):
        failure += 1

accuracy = 1 - failure/total
print("overall accuracy is ",accuracy)


