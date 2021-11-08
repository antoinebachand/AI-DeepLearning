# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 05:28:03 2021

@author: antoi
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets

def load_iris():
    '''
    Load the iris dataset that contains N input features
    of dimension F and N target classes.
    Returns:
    * inputs (np.ndarray): A [N x F] array of input features
    * targets (np.ndarray): A [N,] array of target classes
    '''
    iris = datasets.load_iris()
    return iris.data, iris.target, [0,1,2]

def split_train_test(features: np.ndarray, targets: np.ndarray,
    train_ratio:float=0.8) -> Union[tuple, tuple]:
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    '''
    np.random.seed(123)
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    split_index = int(features.shape[0] * train_ratio)

    train_features, train_targets = features[0:split_index, :],\
    targets[0:split_index]
    test_features, test_targets = features[split_index:-1, :],\
        targets[split_index: -1]

    return (train_features, train_targets), (test_features, test_targets)

def plot_points(points, point_targets):
    '''
    Plot a scatter plot of the first two feature dimensions
    in the point set
    '''
    colors = ['yellow', 'purple', 'blue']
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='black',
            linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

d, t, classes = load_iris()
plt.gcf().set_dpi(300)
plot_points(d, t)


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    ...
    distance =  np.linalg.norm(x - y)
    return distance

def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_list = [points[n] for n in range(len(points))]
    distances = [euclidian_distance(x, i) for i in points_list] 
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    arg = np.argsort(euclidian_distances(x, points))
    return arg[0:k]


from collections import Counter

def vote(targets, classes):  
    return Counter(targets).most_common(1)[0][0]


def knn(x: np.ndarray, points: np.ndarray, point_targets: np.ndarray, classes: list, k: int) -> np.ndarray:
    nea = k_nearest(x, points, k)
    target = [point_targets[i] for i in nea]
    for i in target:
        return vote(target, classes)

         

def remove_one(points: np.ndarray, i: int):
    '''
    Removes the i-th from points and returns
    the new array
    '''
    return np.concatenate((points[0:i], points[i+1:]))


def knn_predict(points: np.ndarray, point_targets: np.ndarray, classes: list, k: int) -> np.ndarray:
    
    rep_list = []
    
    for i in (range(len(points))):
        point_2 = remove_one(points, i)
        target_2 = remove_one(point_targets, i)
        Good_x = points[i,:]
        
        rep = knn(Good_x, point_2, target_2, classes, k)
        rep_list.append(rep)
        
    return rep_list
    
    

from sklearn.metrics import accuracy_score

def knn_accuracy(points: np.ndarray, point_targets: np.ndarray, classes: list, k: int) -> float: 
    test_label = (point_targets)
    predicted = (knn_predict(points, point_targets, classes, k))
    index_list = []
    for i in range(1, (len(points) -1)):
        if predicted[i] != test_label[i]:
           Non = points[i]
           index_list.append(Non)
           x =  np.array(index_list) 
           bad = len(x)
           good = len(points)
           accu = 1 - ((bad/good))
    return accu

def best_k(points: np.ndarray, point_targets: np.ndarray, classes: list) -> int:
    k_list = np.array(range(120))
    accu = [knn_accuracy(points, point_targets, classes, i) for i in k_list ]
    num = np.argmax(accu)
    for i in range(1, (len(points) -1)):
        accuracy = knn_accuracy(points, point_targets, classes, i)
        best_accuracy = max(accu)        
        if accuracy >= best_accuracy:
            best_accuracy=accuracy
            best=i
    return best


from sklearn.metrics import confusion_matrix

def knn_confusion_matrix(points: np.ndarray, point_targets: np.ndarray, classes: list, k: int ) -> np.ndarray:
    test_label = (point_targets)
    predicted = (knn_predict(points, point_targets, classes, k))
    return confusion_matrix(predicted, test_label)

d, t, classes = load_iris()
(d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
print(knn_accuracy(d_train, t_train, classes, 6))
print(best_k(d_train, t_train, classes))
     



def bad_points(points: np.ndarray, point_targets: np.ndarray, classes: list, k: int):
    predicted = (knn_predict(points, point_targets, classes, k))
    test_label = (point_targets)
    index_list = []
    for i in range(len(points)):
        if predicted[i] != test_label[i]:
           Non = points[i]
           index_list.append(Non)
    return np.array(index_list)   
            
#print(bad_points(d, t, classes, 3))
'''
def knn_plot_points(points: np.ndarray, point_targets: np.ndarray, classes: list, k: int):
    colors = ['yellow', 'purple', 'blue']
    for i in range(points.shape[0]):                                                
        [x, y] = points[i,:2]
       
        
        plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='g',linewidths=2)
        plt.scatter(5.9, 3.2, c=colors[point_targets[i]], edgecolors='r',linewidths=2)
        plt.scatter(6.3, 2.5, c=colors[point_targets[i]], edgecolors='r',linewidths=2)
        plt.scatter(6, 2.7, c=colors[point_targets[i]], edgecolors='r',linewidths=2)
        plt.scatter(4.9, 2.5, c=colors[point_targets[i]], edgecolors='r',linewidths=2)
        plt.scatter(6, 2.2, c=colors[point_targets[i]], edgecolors='r',linewidths=2)
        plt.scatter(6.3, 2.8, c=colors[point_targets[i]], edgecolors='r',linewidths=2)
    
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.gcf().set_dpi(300)
    plt.show()
   
knn_plot_points(d, t, classes, 3)       
d, t, classes = load_iris()
print(knn_accuracy(d, t, classes, 5))

x, points = d[0,:], d[1:, :]
x_target, point_targets = t[0], t[1:]
'''
