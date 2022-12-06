import numpy as np 
from scipy.special import expit

def sigmoid(x):
 f = 1/(1 + np.exp(-x))
 return f

g = sigmoid

def my_dense(a_in,W,b):
 units = W.shape[1]
 a_out = np.zeros(units)
 for j in range(units):
  w = W[:,j]
  z = np.dot(w,a_in) + b[j]
  a_out[j] = g(z)
 return a_out

def my_sequential(x,W1,b1,W2,b2):
 a1 = my_dense(x,W1,b1)
 a2 = my_dense(a1,W2,b2)
 return a2


def my_predict(X,W1,b1,W2,b2):
 m = X.shape[0]
 p = np.zeros((m,1))
 for i in range(m):
  p[i,0] = my_sequential(X[i],W1,b1,W2,b2)
 return p


W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )


X_tst = np.array([
    [-0.47,0.42],  # postive example
    [-0.47,3.16]])   # negative example
predictions = my_predict(X_tst, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")
