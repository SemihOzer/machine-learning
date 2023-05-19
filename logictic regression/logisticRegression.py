import pandas as pd
import math
import numpy as np

data = pd.read_csv('diabetes.csv')

outcome = data.pop('Outcome')

y = outcome.to_numpy()
x_train = data.to_numpy()

print(data.head())


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def compute_cost(X, y, w, b, lambda_= 1):
    m, n = X.shape
    cost = 0

    for i in range(m):
        z = np.dot(X[i],w) + b
        f_wb = sigmoid(z)
        cost += -y[i]*np.log(f_wb) - (1- y[i])*np.log(1 - f_wb)
    total_cost = cost / m

    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        err_i = f_wb - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]
        dj_db = dj_db + err_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):

    m = len(X)


    J_history = []
    w_history = []

    for i in range(num_iters):

        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i < 100000:
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history


def predict(X, w, b):

    m, n = X.shape
    p = np.zeros(m)

    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)

        p[i] = f_wb >= 0.5

    return p

w_in = np.zeros_like(x_train[0])
b_in = 0.

w_out, b_out, J_history, w_history = gradient_descent(x_train,y,w_in,b_in,compute_cost,compute_gradient,0.4,700,8)

p = predict(x_train, w_out,b_out)
print(p)
print('Train Accuracy: %f'%(np.mean(p == y) * 100))