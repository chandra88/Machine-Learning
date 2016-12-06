#!/bin/python
# Author: Chandra S. Nepali
# Date: 12-03-2016
# Gradient Descent, Linear regression
# python 2.7

from sympy import Matrix, zeros, ones
import pandas as pd
import matplotlib.pyplot as plt

#-----------------------------------------------------------
def costValue(X, Y, theta):
    m = Y.shape[0]        
    cost = (1.0/(2*m)) * (X*theta - Y).T*(X*theta - Y)
    return cost
#-------------------------
    
def gradientDescent(X, Y, theta, alpha):
    m = Y.shape[0];
    theta = theta - (alpha/m) * X.T*(X*theta - Y)
    return theta
#------------------------

# analytical method
def normalEqu(X, Y):
    theta = zeros(X.shape[1], 1)
    theta = (X.T*X)**-1 * X.T*Y
    return theta
#-----------------------------------------------------------
# -------------- Main function -----------------------------
    
df = pd.read_csv('ex1data1.txt', header=None)
nrows = df.shape[0]
ncols = df.shape[1]

# X-data and its normalization
X = df[[i for i in range(ncols-1)]]
X = X - X.mean(0)
X = X / X.std(0)
#--------------------------

X = Matrix(X)
Y = Matrix(df[[ncols-1]])
theta = zeros(ncols, 1)
X = X.col_insert(0, ones(nrows, 1))

niter = 1500
alpha = 0.01

J_hist = zeros(niter, 1)
theta_hist = zeros(niter, len(theta))

i_hist = zeros(niter, 1)
for i in range(niter):
    cost = costValue(X, Y, theta)
    theta = gradientDescent(X, Y, theta, alpha)
    J_hist[i] = cost[0]
    for j in range(len(theta)):
        theta_hist[i, j] = theta[j, 0]
    i_hist[i] = i

#------------------ plots --------------------------------
# plot cost vs number of interation
plt.figure(1)
plt.plot(list(J_hist)) #, 'k', list(J_hist), 'bo')
plt.xlabel('number of interation')
plt.ylabel('cost')

# plot all theta vs number of interation
j = 1
for i in range(len(theta)):
    j += 1
    plt.figure(j)
    plt.plot(list(theta_hist.col(i)))
    plt.xlabel('number of interation')
    plt.ylabel(r'$\theta$' + str(i))

plt.show()    

#-------------- output -------------------------------
print '---- Gradient desent values of thetas ----'
print '\t', list(theta), '\n'

print '---- analytical values of thetas ----'
theta_norm = normalEqu(X, Y)
print '\t', list(theta_norm), '\n'

print '---- differences ----'
theta = list(theta)
theta_norm = list(theta_norm)
for i in range(len(theta)):
    diff = abs(theta[i] - theta_norm[i])*100.0/abs(theta_norm[i]);
    print '\t', 'diff = ', diff, ' %'
