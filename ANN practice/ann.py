import numpy as np
from process import get_data

path = "ecommerce_data.csv"
X,Y = get_data(path)

M = 5
D = X.shape[1]
K = len(set(Y))

print len(Y)
print K

w1 = np.random.randn(D,M)
b1 = np.zeros(M)

w2 = np.random.randn(M,K)
b2 = np.zeros(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, w1, b1, w2, b2):
    Z = np.tanh(X.dot(w1) + b1)
    return softmax(Z.dot(w2) + b2)

P_Y_given_X = forward(X, w1, b1, w2, b2)
prediction = np.argmax(P_Y_given_X, axis=1)

def classification(Y, P):
    return np.mean(Y == P)

print "Score: ", classification(Y, prediction)


