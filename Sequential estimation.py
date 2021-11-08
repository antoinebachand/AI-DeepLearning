# -*- coding: utf-8 -*-
"""
Antoine TP 3
"""
#=======================================================================
# Tools
#======================================================================= 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def scatter_3d_data(data: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('300 random vectors')    
    plt.gcf().set_dpi(300)
    plt.show()


def bar_per_axis(data: np.ndarray):
    for i in range(data.shape[1]):
        plt.subplot(1, data.shape[1], i+1)
        plt.hist(data[:, i], 100)
        plt.title(f'Dimension {i+1}')     
        plt.gcf().set_dpi(300)
    plt.show()
 

#=======================================================================
# Part 1
#=======================================================================  
      

import matplotlib.pyplot as plt
import numpy as np


import numpy as np


def gen_data(n: int, k: int, mean: np.ndarray, var: float) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''   
    # Sequence of covariance matrices. 
   
    Ident = np.eye(k)
    cov_matrix = Ident * np.square(var)
    # Generate samples from the standard multivariate normal distribution.
    Vect = np.random.multivariate_normal( mean, cov_matrix, n)
    return Vect


#print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
#print(gen_data(5, 1, np.array([0.5]), 0.5))

#=======================================================================
# 1.2 Graph
#======================================================================= 


X = gen_data(300, 3, [0, 1, -1], np.sqrt(3))
#scatter_3d_data(X)
#bar_per_axis(X)

#=======================================================================
# Section 1.4
#======================================================================= 

def update_sequence_mean(mu: np.ndarray, x: np.ndarray, n: int):
    return mu + (x - mu)/n

mean = np.mean(X, 0)
new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
update_sequence_mean(mean, new_x, X.shape[0])

#=======================================================================
# Section 1.5
#======================================================================= 

def _estimates_():
    data = gen_data(100, 3, [0, 0, 0], 1)
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimation_point = update_sequence_mean([0,0,0], data[i], (i+1) )
        estimates.append(estimation_point)
    return estimates

    
def _plot_sequence_estimate():
    estimates = _estimates_()
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.title('Bochang Mean Estimation')
    plt.xlabel('Number of points')
    plt.ylabel('Estimation')
    plt.gcf().set_dpi(600)
    plt.show()
    

_plot_sequence_estimate()
#=======================================================================
# Section 1.6
#======================================================================= 

def _square_error():    
    estimates = _estimates_()
    one_dimension_estimates = [((e[0] + e[1] + e[2]) /3)  for e in estimates]
    del one_dimension_estimates[0]
    y_hat = one_dimension_estimates
    
    Square_error = [ (0 - i)**2 for i in y_hat]
    return Square_error
    
def _plot_mean_square_error():
    #import data
    Square_error = _square_error()
    
    plt.plot(Square_error, label='Square Error')
    plt.legend(loc='upper center')
    plt.title('Mean Sequential Estimation Error')
    plt.xlabel('n')
    plt.ylabel('Error')
    plt.gcf().set_dpi(600)
    plt.show()

_plot_mean_square_error()
#=======================================================================
# BONUS
#======================================================================= 

# Data Generation
Ident = np.eye(1)
cov_matrix = Ident * np.square(1)
# Mean distribution
First_Dimension = np.linspace(0, 1, num=500)
Second_Dimension = np.linspace(1, -1, num=500)
Third_Dimension = np.linspace(-1, 0, num=500)


# Random Vector
Data_1 = [gen_data(1, 1 , [i], 1) for i in First_Dimension]
Data_2 = [gen_data(1, 1 , [i], 1) for i in Second_Dimension]
Data_3 = [gen_data(1, 1 , [i], 1) for i in Third_Dimension]
V1D =Data_1
V2D =Data_2
V3D =Data_3

# Plot Random Vector
def _plot_No_Sequential_():
    
    n = np.linspace(0, 500, num=500)
    plt.plot([e[0] for e in V1D], label='First dimension')
    plt.plot([e[0] for e in V2D], label='Second dimension')
    plt.plot([e[0] for e in V3D], label='Third dimension')
    plt.legend(loc='upper center')
    plt.title('Mean Sequential Estimation')
    plt.xlabel('n')
    plt.ylabel('Mean')
    plt.gcf().set_dpi(600)
    plt.show()

_plot_No_Sequential_()  

#=======================================================================
# Mean sequential estimation
#=======================================================================

def _estimates_1D_():
    data = V1D
    estimates = [np.array([0])]
    for i in range(500):
        estimation_point = update_sequence_mean(First_Dimension[i], data[i], (i+1) )
        estimates.append(estimation_point)
    return estimates

def _estimates_2D_():
    data = V2D
    estimates = [np.array([0])]
    for i in range(500):
        estimation_point = update_sequence_mean(Second_Dimension[i], data[i], (i+1) )
        estimates.append(estimation_point)
    return estimates

def _estimates_3D_():
    data = V2D
    estimates = [np.array([0])]
    for i in range(500):
        estimation_point = update_sequence_mean(Third_Dimension[i], data[i], (i+1) )
        estimates.append(estimation_point)
    return estimates

estimates_1 = _estimates_1D_()
estimates_2 = _estimates_2D_()
estimates_3 = _estimates_3D_()


#=======================================================================
# Plot Changing Mean  
#======================================================================= 

def _plot_No_Forget_():
    
    n = np.linspace(0, 500, num=500)    
    plt.plot([e[0] for e in estimates_1], label='First dimension')
    plt.plot([e[0] for e in estimates_2], label='Second dimension')
    plt.plot([e[0] for e in estimates_3], label='Third dimension')
    plt.legend(loc='upper center')
    plt.title('Mean Sequential Estimation')
    plt.xlabel('n')
    plt.ylabel('Mean')
    plt.gcf().set_dpi(600)
    plt.show()

_plot_No_Forget_()  

#=======================================================================
# Square error  
#=======================================================================

def _square_error_1():   
    mean = First_Dimension
    y_hat =  estimates_1
    Square_error = [ (mean[i] - y_hat[i])**2 for i in range(500)]
    return Square_error
def _square_error_2():   
    mean = Second_Dimension
    y_hat =  estimates_2
    Square_error = [ (mean[i] - y_hat[i])**2 for i in range(500)]
    return Square_error
def _square_error_3():   
    mean = Third_Dimension
    y_hat =  estimates_3
    Square_error = [ (mean[i] - y_hat[i])**2 for i in range(500)]
    return Square_error

def _total_square_error():
    E1 = _square_error_1()
    E2 = _square_error_2()
    E3 = _square_error_3()
    Total_error = [(E1[i] + E2[i] + E3[i]) for i in range(len(E1))]
    return Total_error

    
def _plot_mean_square_error():
    #import data
    total_E = _total_square_error()
    
    plt.plot(total_E, label='Square Error')
    plt.legend(loc='upper center')
    plt.title('Mean Sequential Estimation Error')
    plt.xlabel('n')
    plt.ylabel('Error')
    plt.gcf().set_dpi(600)
    plt.show()

_plot_mean_square_error()








