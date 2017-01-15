# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from util import getData, getBinaryData, plot_examples
from ANN import ANN

def classify_binary(dataset_path):
    X, Y   = getBinaryData(dataset_path)
    X0, X1 = X[Y==0,:], X[Y==1,:]
    X1 = np.repeat(X1, 9, axis = 0) #duplicate the data 9x
    X, Y   = np.vstack([X0, X1]), np.array([0]*len(X0) + [1]*len(X1))
    
    #this is for classify a binary classification
    model = ANN(100)
    model.fit_2class(X, Y, show_fig=True)
    model.score(X, Y)

def classify_multiclass(dataset_path):
    X, Y   = getData(dataset_path)
    
    #this is for classify multiclass classification
    model = ANN(200)
    model.fit(X, Y, show_fig=True)
    print model.score(X, Y)
    
    
if __name__ == "__main__":
#    plot_examples(dataset_path)
    dataset_path = "C:/Users/Brilian/Documents/Python Scripts/fer2013"
#    classify_binary(dataset_path)
    classify_multiclass(dataset_path)
    