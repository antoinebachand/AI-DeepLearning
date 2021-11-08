from typing import Union

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
    return iris.data, iris.target, [0, 1, 2]


def split_train_test(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.8
) -> Union[tuple, tuple]:
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    '''
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    split_index = int(features.shape[0] * train_ratio)

    train_features, train_targets = features[0:split_index, :],\
        targets[0:split_index]
    test_features, test_targets = features[split_index:-1, :],\
        targets[split_index: -1]

    return (train_features, train_targets), (test_features, test_targets)

#============================================================
#Section 1
#============================================================

import matplotlib.pyplot as plt
from typing import Union
import numpy as np



features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)

def sigmoid(x: float) -> float:
    if x< -100:
        return 0.0
    else:
        return 1/(1+np.exp(-x))

def d_sigmoid(x: float) -> float:
    return sigmoid(x)*(1-sigmoid(x))


def perceptron(x: np.ndarray,w: np.ndarray) -> Union[float, float]:
    w_sum = np.sum(x*w)
    sig = sigmoid(w_sum)
    return (w_sum, sig)


def ffnn(x: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray,) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
     
    z0 = np.insert(x, 0, 1.0)
    z1 = np.zeros(M)
    a1 = np.zeros(M)

    a2 = np.zeros(K)
    y = np.zeros(K)

    for m in range(M):
        a1[m], z1[m] = perceptron(z0, W1[:, m])

    z1 = np.insert(z1, 0, 1.0)

    for k in range(K):
        a2[k], y[k] = perceptron(z1, W2[:, k])
    
    
    return y, z0, z1, a1, a2

    
# x = train_features[0, :]
# K = 3 # number of classes
# M = 10
# D=len(x)

# W1 = 2 * np.random.rand(D + 1, M) - 1
# W2 = 2 * np.random.rand(M + 1, K) - 1
# y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2) 
# print(ffnn(x, M, K, W1, W2)[0] )


def backprop(x: np.ndarray, target_y: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''

    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    delta_k = y - target_y
    delta_j = []

    for a in range(len(a1)):
        delta_j.append(d_sigmoid(a1[a]) * (np.sum(W2[a+1] * delta_k)))
    
    dE1, dE2 = np.zeros(W1.shape), np.zeros(W2.shape)
    
    for j in range(len(delta_j)):
        for z in range(len(z0)):
            dE1[z][j] = delta_j[j] * z0[z]
    
    for k in range(len(delta_k)):
        for z in range(len(z1)):
            dE2[z][k] = delta_k[k] * z1[z]
    
    return y, dE1, dE2





K = 3  # number of classes
M = 6
D = train_features.shape[1]

x = features[0, :]

# create one-hot target for the feature
target_y = np.zeros(K)
target_y[targets[0]] = 1.0

np.random.seed(42)
#Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1

y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)

#============================================================
#Section 2: Training and test Data
#============================================================

def train_nn(X_train: np.ndarray, t_train: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray, iterations: int, eta: float) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    N = len(X_train)
    guesses = [0] * N
    
    misclassification_rate, Etotal = [], []
    
    for i in range(iterations):
        dE1_total, dE2_total = np.zeros(W1.shape), np.zeros(W2.shape)    
        errors, misclass = 0, 0
        
        for n in range(N):
            target_y = np.zeros(K)
            target_y[t_train[n]] = 1.0
            y, dE1, dE2 = backprop(X_train[n], target_y, M, K, W1, W2)
            dE1_total += dE1
            dE2_total += dE2

            guesses[n] = np.argmax(y)
            errors += ((target_y * np.log(np.array(y)))
                    + ((1 - target_y) * np.log( 1 - np.array(y))))
            
            if np.argmax(target_y) != guesses[n]:
                misclass += 1
       
        W1 -= eta * dE1_total / N
        W2 -= eta * dE2_total / N

        Etotal.append(np.sum(-errors) / N)
        misclassification_rate.append(misclass / N)
        
    plt.plot(range(iterations),Etotal, color='g')
    plt.xlabel('Iterations')
    plt.title('E total as a function of iterations')
    plt.ylabel('Etotal')
    plt.gcf().set_dpi(600)
    plt.show()
       
    plt.plot(range(iterations),misclassification_rate, color='orange')
    plt.xlabel('iterations')
    plt.title('Misclassification rate as a function of iterations')
    plt.ylabel('misclassification rate')
    plt.gcf().set_dpi(600)
    plt.show() 
    
    return W1, W2, Etotal, misclassification_rate, guesses
    
K = 3  # number of classes
M = 6
D = train_features.shape[1]
np.random.seed(42)
# Initialize two random weight matrices
W1 = 2 * np.random.rand(D + 1, M) - 1
W2 = 2 * np.random.rand(M + 1, K) - 1
W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
    train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.3)    

def test_nn(X: np.ndarray, M: int, K: int, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    guesses = []
    for x in X:
        y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
        guesses.append(np.argmax(y))
    return guesses

#============================================================
#Section 3: Confusion matrix and accuracy
#============================================================
features, targets, classes = load_iris()
(train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)

x = train_features[0,:]
K = 3
M = 10

W1 = 2*np.random.rand(len(x) + 1, M) - 1
W2 = 2*np.random.rand(M +1, K) -1
y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
print(y, z0, z1, a1, a2)

K = len(classes)
M = 6
D = train_features.shape[1]
x = features[0,:]

target_y = np.zeros(K)
target_y[targets[0]] = 1.0

np.random.seed(42)

W1 = 2*np.random.rand(D + 1, M) -1
W2 = 2* np.random.rand(M +1, K) - 1

y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
print(y, dE1, dE2)

K = len(classes)
M = 6
D = train_features.shape[1]
np.random.seed(42)

iterations = 500

W1tr, W2tr, Etotal, misclass, last_g, = train_nn(train_features[:20,:], train_targets[:20], M, K, W1, W2, iterations, 0.3)
print(W1tr, W2tr, Etotal, misclass, last_g)

guesses = test_nn(test_features, M, K, W1tr, W2tr)
print(guesses)


    

# 1. Calculate the accuracy      
from sklearn.metrics import accuracy_score
print(accuracy_score(test_targets, guesses, normalize=True))

# 2. produce a confusion matrix for your test features and test predictions
from sklearn.metrics import confusion_matrix 
print(confusion_matrix(test_targets,guesses))

# 3. Plot the E_total as a function of iterations from your train_nn function.
plt.figure(1)
plt.plot(range(iterations),Etotal)
plt.title('Etotal')
plt.xlabel('iterations')
plt.ylabel('Etotal')
plt.show()
# 4. Plot the misclassification_rate as a function of iterations from your train_nn function.
plt.figure(2)
plt.plot(range(iterations),misclass)
plt.title('Misclassification rate')
plt.xlabel('iterations')
plt.ylabel('Misclassification')
plt.show()
