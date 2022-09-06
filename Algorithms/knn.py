# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 09:26:52 2022

@author: 19525
"""

"""
Notes on Knn

First we want to have a normalized dataset for the X values 

Next we want to build a function that brings in the training set seperated
by predictors and label array, and the testing set of predictors and k values

Next we want to 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def normalize(array):
    max_arr = np.max(array)
    min_arr = np.min(array)
    range_arr = max_arr - min_arr
    
    norm = []
    for i in range(array.size):
        norm.append((array[i] - min_arr) / range_arr)
        
    return norm


iris = pd.read_csv('iris.csv')

iris = iris[iris['class'] != 'Iris-setosa'].reset_index(drop = True)


X_cols = len(iris.columns)-1
X_norm = iris.iloc[:, :X_cols].apply(lambda x: normalize(x), axis = 0)
y = iris['class']

X_train_norm, X_test_norm, y_train, y_test = train_test_split(
    X_norm, y, test_size = 0.3, random_state = 123)

X_val_norm, X_test_norm, y_val, y_test = train_test_split(
    X_test_norm, y_test, test_size = 1/3, random_state = 123)



# distance can be manh, eucl, maxcoord
# knn_type can class or regr

class_1 = 'Iris-virginica'
class_2 = 'Iris-versicolor'
k = 5
dist = 'man'

testX = X_val_norm
trainX = X_train_norm

trainY = y_train

    
def knnClass(trainX, trainY, testX, k, dist, class_1, class_2):
#     """
#     Notes on knnClass function:
#
#     Only accepts binomial classification
#
#     Inputs:
#         trainX: the training set predictors
#         trainY: the training label
#         testX: the test set predictors
#         k: hyperparameter to tune for number of datapoints in trainX closest to testX value
#         dist: distance formula to use. Accepted values are eucl for euclidean distance,
#               man for manhattan distnace, and maxc for max coordinate distance
#         class_1 = input the name of the first class
#         class_2 = input for the name of the second class
#
#     Outputs: Two lists with probability predictions for each class
# 
#     """

    if dist == 'eucl':
        class1_pr_list = []
        class2_pr_list = []
        
        for i in range(len(testX)):
            list_val = []
            for j in range(len(trainX)):
                value = np.sqrt(np.sum(np.square(trainX.iloc[j,] - testX.iloc[i,])))
                list_val.append(value)
                
            list_val = np.array(list_val)
            top_k = trainY.to_numpy()[np.argsort(list_val)][:k]

            pr1 = np.sum(top_k == class_1) / top_k.size
            pr2 = np.sum(top_k == class_2) / top_k.size
            
            class1_pr_list.append(pr1)
            class2_pr_list.append(pr2)
            
    elif dist == 'man':
        class1_pr_list = []
        class2_pr_list = []
        
        for i in range(len(testX)):
            list_val = []
            for j in range(len(trainX)):
                value = np.sum(np.abs(trainX.iloc[j,] - testX.iloc[i,]))
                list_val.append(value)
                
            list_val = np.array(list_val)
            top_k = trainY.to_numpy()[np.argsort(list_val)][:k]

            pr1 = np.sum(top_k == class_1) / top_k.size
            pr2 = np.sum(top_k == class_2) / top_k.size
            
            class1_pr_list.append(pr1)
            class2_pr_list.append(pr2) 
            
    elif dist == 'maxc':
        class1_pr_list = []
        class2_pr_list = []
        
        for i in range(len(testX)):
            list_val = []
            for j in range(len(trainX)):
                value = np.abs(trainX.iloc[j,] - testX.iloc[i,])
                list_val.append(value)
            
            list_val = np.array(list_val)
            top_k = trainY.to_numpy()[np.argsort(list_val)][:k]

            pr1 = np.sum(top_k == class_1) / top_k.size
            pr2 = np.sum(top_k == class_2) / top_k.size
            
            class1_pr_list.append(pr1)
            class2_pr_list.append(pr2)
            
    return class1_pr_list, class2_pr_list


pr1, pr2 = knnClass(trainX, trainY, testX, k, dist, class_1, class_2)


def knnRegr(trainX, trainY, testX, k, dist):
#     """
#     Notes on knnRegr function:
#
#     Only accepts binomial classification
#
#     Inputs:
#         trainX: the training set predictors
#         trainY: the training label
#         testX: the test set predictors
#         k: hyperparameter to tune for number of datapoints in trainX closest to testX value
#         dist: distance formula to use. Accepted values are eucl for euclidean distance,
#               man for manhattan distnace, and maxc for max coordinate distance
#         class_1 = input the name of the first class
#         class_2 = input for the name of the second class
#
#     Outputs: one lists with prediction values
# 
#     """

    if dist == 'eucl':
        avg_list = []
        
        for i in range(len(testX)):
            list_val = []
            for j in range(len(trainX)):
                value = np.sqrt(np.sum(np.square(trainX.iloc[j,] - testX.iloc[i,])))
                list_val.append(value)
                
            list_val = np.array(list_val)
            top_k = trainY.to_numpy()[np.argsort(list_val)][:k]

            avg = np.sum(top_k) / top_k.size
            
            avg_list.append(avg)
            
            
    elif dist == 'man':
        avg_list = []
        
        for i in range(len(testX)):
            list_val = []
            for j in range(len(trainX)):
                value = np.abs(trainX.iloc[j,] - testX.iloc[i,])
                list_val.append(value)
            
            list_val = np.array(list_val)
            top_k = trainY.to_numpy()[np.argsort(list_val)][:k]

            avg = np.sum(top_k) / top_k.size
            
            avg_list.append(avg)
            
            
    return avg_list


    
  