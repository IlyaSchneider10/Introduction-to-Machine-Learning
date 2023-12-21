# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic, Product, Sum, ExpSineSquared
from sklearn.model_selection import KFold

# We defeine a function to impute the missing data

def impute_data(data):

    split = data.groupby(data.loc[:,"season"])

    imputer = KNNImputer()
    imputed_data = []
    
    groups = ["spring", "summer", "autumn", "winter"]
    for group in groups:

        full_group = split.get_group(group)
        season = full_group.loc[:,"season"] 
        group_to_impute = full_group.drop(['season'],axis=1)
        imputed_group = imputer.fit_transform(group_to_impute)
        imputed_group = pd.DataFrame(imputed_group, columns = group_to_impute.columns, index =  season.index)
        output_group = pd.concat([season, imputed_group], axis=1) 
        imputed_data.append(output_group)

            
    imputed_data = pd.concat(imputed_data)
    imputed_data = imputed_data.sort_index()

    return imputed_data

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Data preprocessing, imputation and extract X_train, y_train and X_test
    imputed_train = impute_data(train_df)
    imputed_test = impute_data(test_df)

    # Substitute seasons for values: 0 or 1

    encoder = OneHotEncoder(sparse=False)
    encoded_seasons = encoder.fit_transform(imputed_train['season'].values.reshape(-1, 1))
    encoded_seasons_test = encoder.transform(imputed_test['season'].values.reshape(-1, 1))
    encoded_seasons_df = pd.DataFrame(encoded_seasons, columns=encoder.get_feature_names_out(['season']))
    encoded_seasons_df_test = pd.DataFrame(encoded_seasons_test, columns=encoder.get_feature_names_out(['season']))
    imputed_train = pd.concat([encoded_seasons_df, imputed_train.drop('season', axis=1)], axis=1)
    imputed_test = pd.concat([encoded_seasons_df_test, imputed_test.drop('season', axis=1)], axis=1)

    X_train = imputed_train.drop(['price_CHF'],axis=1).to_numpy()
    y_train = imputed_train['price_CHF'].to_numpy()
    X_test = imputed_test.to_numpy()

    # scaler = StandardScaler()
    # X_train =  scaler.fit_transform(X_train)
    # X_test=  scaler.transform(X_test)

    #assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

# Functions for picking the best kernel by implementing cross validation

def calculate_RMSE(y_truth, y_pred):
    
    RMSE=np.sqrt(np.sum((y_truth-y_pred)**2)/y_truth.shape[0])

    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X, y, kernels, n_folds):

    RMSE_mat = np.zeros((n_folds, len(kernels)))        
    kf=KFold(n_folds)

    for kf_i, (train_index, test_index) in enumerate(kf.split(X_train)):
        
        train_data=X_train[train_index,:]
        train_labels=y_train[train_index]
        test_data=X_train[test_index,:]
        test_labels=y_train[test_index]

        for k_i, k in enumerate(kernels):

            gpr = GaussianProcessRegressor(kernel=k)
            gpr.fit(train_data, train_labels)

            y_pred=gpr.predict(test_data)
            run_RMSE=calculate_RMSE(test_labels, y_pred)

            RMSE_mat[kf_i,k_i]=run_RMSE
    
    avg_RMSE = np.mean(RMSE_mat, axis=0)
    
    assert avg_RMSE.shape == (len(kernels),)
    return avg_RMSE

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    #TODO: Define the model and fit it using training data. Then, use test data to make predictions 
    kernels = [DotProduct(), RBF(), Matern(), RationalQuadratic(), Product(DotProduct(),RationalQuadratic()), Sum(RBF(),Matern()), Sum(DotProduct(), RationalQuadratic()), Sum(RBF(),RationalQuadratic()), Sum(Matern(),RationalQuadratic())]
    n_folds = 10

    kernel_summary = average_LR_RMSE(X_train, y_train, kernels, n_folds)

    best_kernel = kernels[np.argmin(kernel_summary)]

    gpr = GaussianProcessRegressor(kernel=best_kernel)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results_5.csv', index=False)
    print("\nResults file successfully generated!")

