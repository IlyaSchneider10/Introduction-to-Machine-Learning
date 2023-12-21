# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold

training_file = pd.read_csv("train.csv",index_col=0)
training_file=training_file.to_numpy()
training_data=training_file[:,1:]
training_labels=training_file[:,0]
    
    
def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """

    X_transformed = np.zeros((X.shape[0], 21))
    
    X_transformed[:,0:5]=X
    X_transformed[:,5:10]= X**2  
    X_transformed[:,10:15]=np.exp(X)
    X_transformed[:,15:20]=np.cos(X)
    X_transformed[:,20]=1.
        
    assert X_transformed.shape == (X.shape[0], 21)
    return X_transformed

def fit(X, y, lam):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    X_transformed = transform_data(X)
    w=np.zeros((21,))
    reg = linear_model.Ridge(alpha=lam, fit_intercept=False)
    reg.fit(X_transformed, y)
    w=reg.coef_
    
    assert w.shape == (21,)
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
            run_RMSE=calculate_RMSE(weights,transform_data(test_data), test_labels)
            
        
            RMSE_mat[k_i,l_i]=run_RMSE
    
    avg_RMSE = np.mean(RMSE_mat, axis=0)
    
    #assert avg_RMSE.shape == (5,)
    return avg_RMSE

lambdas = [x*0.1 for x in range(10,1011)]
n_folds = 10
avg_RMSE = average_LR_RMSE(training_data, training_labels, lambdas, n_folds)

best_lambda=lambdas[np.argmin(avg_RMSE)]

final_w = fit(training_data, training_labels, best_lambda)
np.savetxt("./results_ilya.csv", final_w, fmt="%.12f")

