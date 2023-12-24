#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# Part 1 - Implement k-Nearest Neighbours (kNN) - 30 points

def distance(x1, x2):
    distance = np.sqrt((np.square(x1-x2)).sum(axis = -1))
    return distance
  
class kNNClassifier:
  def __init__(self, n_neighbors, X, y):
    self.n_neighbors = n_neighbors
    self.X = X
    self.y = y

  def fit(self, X, y):
    return np.array(X),np.array(y)

  def predict(self, X):
    X_train, y_train = self.fit(self.X,self.y)
    X = np.array(X)
    m, n = X_train.shape
    k, l = X.shape
    if n != l:
      print('dimension of X_train and X_test do not match')
      return -1

    X_train = X_train*np.ones((k,m,n))
    X = X.reshape(k,1,n)

    distance_i = distance(X_train,X)
    distance_i_copy = distance_i.copy()

    k_nearest = np.ones((self.n_neighbors,k))
    for j in range(self.n_neighbors):
      k_i = np.argmin(distance_i_copy,axis=-1)
      distance_i_copy[[np.arange(k)],[k_i]]= distance_i_copy.max(axis=-1)+1
      k_nearest[j] = k_i

    k_nearest = k_nearest.T.astype(int)
    y_test = 5*np.ones(k)
    for i in range(k):
      vals, counts = np.unique(y_train[k_nearest[i]], return_counts=True)
      y_test[i] =  vals[np.argmax(counts)]
    return y_test

X_train = np.arange(15).reshape((5,3))
y_train = [0,0,1,1,1]
X = np.arange(12).reshape((4,3))
model =  kNNClassifier(n_neighbors=3, X=X_train, y=y_train)
print(model.predict(X))

