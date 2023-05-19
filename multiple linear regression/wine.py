import copy
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("winequality-red.csv")
quality = data.pop("quality")
y = quality.to_numpy()
x_train = data.to_numpy()



def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost



def compute_gradient(X,y,w,b):
    m,n = x_train.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i],w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X,y,w,b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(X,y,w,b))

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w,b,J_history



init_w = np.zeros(x_train.shape[1])
init_b = 0.

iterations = 1590
alpha = 0.0002

w_final, b_final, J_hist = gradient_descent(x_train, y, init_w, init_b,
                                                    compute_cost, compute_gradient,
                                                    alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m = x_train.shape[0]
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y[i]}, {y[i] - (np.dot(x_train[i], w_final) + b_final):0.2f}")