# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

def fit(X, y, lam):
    
    #fit the data
    ridge=Ridge(lam, fit_intercept=False)
    ridge.fit(X,y)
   
    #Return parameters
    w =  ridge.coef_
    assert w.shape == (13,)
    return w


def calculate_RMSE(w, X, y):
    
    predicted_labels=np.matmul(X,w)
    RMSE=np.sqrt(np.sum((y-predicted_labels)**2)/y.shape[0])

    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X, y, lambdas, n_folds):
    
    RMSE_mat = np.zeros((n_folds, len(lambdas)))        
    kf=KFold(n_folds)
    
    for k_i, (train_index, test_index) in enumerate(kf.split(X)):
        
        train_data=X[train_index,:]
        train_labels=y[train_index]
        test_data=X[test_index,:]
        test_labels=y[test_index]
        
        for l_i, l in enumerate(lambdas):
        
            weights=fit(train_data,train_labels,l)
            run_RMSE=calculate_RMSE(weights, test_data, test_labels)
        
            RMSE_mat[k_i,l_i]=run_RMSE
    
    avg_RMSE = np.mean(RMSE_mat, axis=0)
    
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results_ilya.csv", avg_RMSE, fmt="%.12f")
