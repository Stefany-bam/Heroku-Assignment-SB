# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 20:29:35 2022

@author: Stefany
"""

#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('NSTD_Assignment.csv')

X = dataset.iloc[:, :2]

def convert_to_int(word):
    word_dict = {'male':1, 'female':2}
    return word_dict[word]

X['Gender'] = X['Gender'].apply(lambda x : convert_to_int(x))
y = dataset.iloc[:, -1]

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X, y)

pickle.dump(knn, open('modelSB.pkl','wb'))
model = pickle.load(open('modelSB.pkl','rb'))

print(model.predict([[2,15]]))