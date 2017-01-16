# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 20:38:55 2017

@author: Brilian
"""

from util import sigmoid, sigmoid_cost, error_rate, relu, cost, softmax, y2indicator
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

class ANN(object):
    def __init__(self, M): #M is the number of hidden unit
        self.M = M
    
    def generate_w_b(self, N, D, totClass = 0, multi=False):
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.b1 = np.zeros(self.M)
        if multi == True:
            self.W2 = np.random.randn(self.M, totClass) / np.sqrt(self.M + totClass)
            self.b2 = np.zeros(totClass)
        else:
            self.W2 = np.random.randn(self.M) / np.sqrt(self.M)
            self.b2 = 0
            
    def forward(self, X):
#        Z = relu(X.dot(self.W1) + self.b1)
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return sigmoid(Z.dot(self.W2) + self.b2), Z
    
    def predict(self, X):
        pY, _ = self.forward(X)
        return np.round(pY)
    
    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)
        
    def show_fig_cost(self, costs, show_fig):
        if show_fig:
            plt.plot(costs)
            plt.show()
            
    def prepare_data(self, X, Y, multi=False):
        #divide into train and test data
        X, Y = shuffle(X, Y)
        Xtest, Ytest = X[-1000:], Y[-1000:]
        Xtrain, Ytrain = X[:-1000], Y[:-1000]
        N, D = Xtrain.shape
        lenY = len(set(Y))
        self.generate_w_b(N, D, lenY, multi=multi)
        return Xtest, Ytest, Xtrain, Ytrain
        
    def fit_2class(self, X, Y, learning_rate = 5*10e-7, \
            reg=1.0, epoch = 10000, show_fig = False):
        Xtest, Ytest, Xtrain, Ytrain = self.prepare_data(X, Y)
        
        costs = []
        best_validation_error = 1
        for i in xrange(epoch):
            pY, Z = self.forward(Xtrain) #forward prop
            
            #back prop
            pY_Y = pY - Ytrain
            self.W2 -= learning_rate * (Z.T.dot(pY_Y) + reg*self.W2)
            self.b2 -= learning_rate * (pY_Y.sum() + reg*self.b2)
            
#            dZ = np.outer(pY_Y, self.W2)* (Z > 0) #Z > 0 is derivative of ReLU
            dZ = np.outer(pY_Y, self.W2)* (1 - Z*Z)
            self.W1 -= learning_rate * (Xtrain.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate * (np.sum(dZ, axis = 0) + reg*self.b1)
            
            if i % 20 == 0:
                pYtest, _ = self.forward(Xtest)
                c = sigmoid_cost(Ytest, pYtest)
                costs.append(c)
                e = error_rate(Ytest, np.round(pYtest))
                print "i: ", i, "cost: ", c, "error: ", e
                if e < best_validation_error: best_validation_error = e
#                if e > best_validation_error: learning_rate /= 2
        print "best validation error:", best_validation_error
        self.show_fig_cost(costs, show_fig)
         
    def forward_multi(self, X):
#        Z = relu(X.dot(self.W1) + self.b1)
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2), Z
    
    def predict_multi(self, X):
        pY, _ = self.forward(X)
        return np.argmax(pY, axis=1)
        
            
    def fit(self, X, Y, learning_rate = 10e-7, \
            reg=10e-7, epoch = 10000, show_fig = False):
        #divide into train and test data
        Xtest, Ytest, Xtrain, Ytrain = self.prepare_data(X, Y, multi=True)
        Ttrain = y2indicator(Ytrain)
        Ttest  = y2indicator(Ytest)
        
        costs = []
        best_validation_error = 1
        for i in xrange(epoch):
            pY, Z = self.forward_multi(Xtrain) #forward prop
            
            #back prop
            pY_Y = pY - Ttrain
            self.W2 -= learning_rate * (Z.T.dot(pY_Y) + reg*self.W2)
            self.b2 -= learning_rate * (pY_Y.sum(axis=0) + reg*self.b2)
            
#            dZ = np.outer(pY_Y, self.W2)* (Z > 0) #Z > 0 is derivative of ReLU
#            print pY_Y.shape, self.W2.shape, Z.shape
            dZ = pY_Y.dot(self.W2.T) * (1 - Z*Z)
            self.W1 -= learning_rate * (Xtrain.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate * (np.sum(dZ, axis = 0) + reg*self.b1)
            
            if i % 10 == 0:
                pYtest, _ = self.forward_multi(Xtest)
                c = cost(Ttest, pYtest)
                costs.append(c)
                e = error_rate(Ytest, np.argmax(pYtest, axis = 1))
                print "i: ", i, "cost: ", c, "error: ", e
                if e < best_validation_error: best_validation_error = e
        print "best validation error:", best_validation_error
        self.show_fig_cost(costs, show_fig)