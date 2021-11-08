# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 05:02:03 2021

@author: antoi

Support Vector Machine with the breast cancer  dataset of sklearn
"""


import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split


def load_cancer():
    '''
    Return the breast cancer dataset split into
    train and test sets.
    '''
    cancer = datasets.load_breast_cancer()
    X_train, X_test, t_train, t_test = train_test_split(
        cancer.data, cancer.target,
        test_size=0.3)
    return (X_train, t_train), (X_test, t_test)


def load_binary_iris():
    '''
    Load the iris dataset that contains N input features
    of dimension F and N target classes. Only load classes
    0 an 1.
    Returns:
    * inputs (np.ndarray): A [N x F] array of input features
    * targets (np.ndarray): A [N,] array of target classes
    '''
    iris = datasets.load_iris()
    index = np.hstack((
        np.where(iris.target == 0),
        np.where(iris.target == 1)))
    iris.data = iris.data[index[0], :]
    iris.target = iris.target[index[0]]
    X_train, X_test, t_train, t_test = train_test_split(
        iris.data, iris.target,
        test_size=0.3)
    return (X_train, t_train), (X_test, t_test)


def plot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray
):
    '''
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.
    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    '''
    plt.scatter(X[:, 0], X[:, 1], c=t, s=30,
                cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svc.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z,
        colors='k', levels=[-1, 0, 1],
        alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.title('Support Vector')
    plt.xlabel('X')
    plt.ylabel('t')
    plt.gcf().set_dpi(300)
    plt.show()
    
#============================================================
#Section 1
#============================================================


from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import (accuracy_score, precision_score, recall_score)
                            

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

#============================================================
#Section 1.1
#============================================================
def _plot_linear_kernel():
    X, t = make_blobs(n_samples = 40, centers = 2)
    print(X.shape)
    clf = svm.SVC(kernel = 'linear', C=1000)
    clf.fit(X, t)
    plot_svm_margin(clf, X, t)
  
_plot_linear_kernel()

#============================================================
#Section 1.2
#============================================================

def _subplot_svm_margin(svc, X: np.ndarray, t: np.ndarray, num_plots: int, index: int):

    plt.subplot(1,num_plots,index)
    plt.scatter(X[:, 0], X[:, 1], c=t, s=20,cmap=plt.cm.Paired)
    
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svc.decision_function(xy).reshape(XX.shape)
    
    # plot support vectors
    ax.contour(XX, YY, Z,colors='k', levels=[-1, 0, 1],alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(svc.support_vectors_[:, 0],svc.support_vectors_[:, 1],s=100, linewidth=1, facecolors='none', edgecolors='k')
    
    # Quality and title 
    
    plt.gcf().set_dpi(300)
    plt.suptitle('Linear support vector with C value of 1000, 0.5, 0.3, 0.05, 0.0001',fontsize=10)
    plt.xlabel('X')
    plt.ylabel('t')
    
    
def _compare_gamma():
    X, t = make_blobs(n_samples=40, centers=2, random_state= 6)

    gamma_list = ['auto', 0.2, 2]

    for i in range(len(gamma_list)):
        clf = svm.SVC(kernel = 'rbf', gamma=gamma_list[i], C = 1000)
        clf.fit(X, t)
        _subplot_svm_margin(clf, X, t, 3, i+1)
        
    plt.show()
    
_compare_gamma()   

#============================================================
#Section 1.5
#============================================================
    
def _compare_C():
    X, t = make_blobs(n_samples=40, centers=2, n_features=2,random_state=0)
    C_list = [1000, 0.5, 0.3, 0.05, 0.0001]  
    
    for i in range(len(C_list)):
        
        clf = svm.SVC(kernel = 'linear', C = C_list[i])
        clf.fit(X, t)
        _subplot_svm_margin(clf, X, t, 5, i+1)
    plt.show()

_compare_C()

#============================================================
#Section 2
#============================================================

def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
  
    svc = svc.fit(X_train, t_train)
    vector = svc.predict(X_test)

    train_test_SVM = (accuracy_score(t_test, vector),precision_score(t_test, vector),recall_score(t_test, vector))
    return train_test_SVM


(X_train, t_train), (X_test, t_test) = load_cancer()
svc = svm.SVC(C=1000)
print(train_test_SVM(svc, X_train, t_train, X_test, t_test))



def compare_SVM():
    (X_train, t_train), (X_test, t_test) = load_cancer()

    svc = svm.SVC(kernel = 'linear', C=1000).fit(X_train, t_train)   
    test_SVM_linear = train_test_SVM(svc, X_train, t_train, X_test, t_test)
    
    svc = svm.SVC(kernel = 'rbf', gamma='auto').fit(X_train, t_train)
    test_SVM_radial = train_test_SVM(svc, X_train, t_train, X_test, t_test)

    svc = svm.SVC(kernel = 'poly', degree=3).fit(X_train, t_train)
    test_SVM_polyn = train_test_SVM(svc, X_train, t_train, X_test, t_test)
    
    print('train_test_SVM_linear: {}'.format(test_SVM_linear))
    print('train_test_SVM_radial: {}'.format(test_SVM_radial))
    print('train_test_SVM_polyn: {}'.format(test_SVM_polyn))


compare_SVM()

