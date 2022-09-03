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
class_num = 2
class_1 = 'Iris-virginica'
class_2 = 'Iris-versicolor'
k = 5
knn_type = 'class'
distance = 'eucl'
    

if class_num == 2:
    if distance == 'eucl':
        if knn_type == 'class':
            class1_pr_list = []
            class2_pr_list = []
            
            for i in range(len(X_val_norm)):
                list_val = []
                for j in range(len(X_train_norm)):
                    value = np.sqrt(np.sum(np.square(X_train_norm.iloc[j,] - X_val_norm.iloc[i,])))
                    list_val.append(value)
                    
                list_val = np.array(list_val)
                top_k = y_train.to_numpy()[np.argsort(list_val)][:k]
                print(top_k)

                pr1 = np.sum(top_k == class_1) / top_k.size
                pr2 = np.sum(top_k == class_2) / top_k.size
                
                class1_pr_list.append(pr1)
                class2_pr_list.append(pr2)





# def knn(trainX, trainY, testX, k = 2, type = 'class'):
#     """
#     Notes on Knn

#     First we want to have a normalized dataset for the X values 

#     Next we want to build a function that brings in the training set seperated
#     by predictors and label array, and the testing set of predictors and k values

#     Next we want to 
#     """
    
a
    