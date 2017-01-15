import numpy as np
import pandas as pd

def get_data(loc):
    df = pd.read_csv(loc) #read the csv
    data = df.as_matrix() #turn the csv into matrix (easy to process)
    X = data[:, :-1] #split the data to last column-1
    Y = data[:,-1] #get the last column (the classes)
    
    #normalize the data
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()
    
    #work on the category column
    N, D = X.shape #get the row and col
    X2 = np.zeros((N,D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
    
    for n in range(N):
        t = int(X[n,D-1])
        X2[n,t + D - 1] = 1 #make the time of day into categorical
    
    #faster method to set the categorical without loop
    #Z = np.zeros((N,4)) 
    #Z[np.arange(N),X[:,D-1].astype(np.int32)] = 1
    #X2[:,-4:] = Z
    #assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)
    return X2, Y

# def get_binary_data(path):
#     X, Y = get_data(path)
#     X2 = X[Y <= 1]
#     Y2 = Y[Y <= 1]
#     return X2, Y2