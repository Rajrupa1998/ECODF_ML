import pandas as pd
import numpy as np
import time
from pyJoules.handler.pandas_handler import PandasHandler
from sklearn.linear_model import LinearRegression
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge



pandas_handler = PandasHandler()

def linear_regression(X_train,Y_train):
    
    linear_regression_model = LinearRegression()
    return linear_regression_model.fit(X_train,Y_train)

@measure_energy(handler=pandas_handler)
def test_linear_regression(X_train,Y_train):
     
     linear_regression(X_train,Y_train)

def store_values():
    df = pandas_handler.get_dataframe()
    return df

#linear_regression_model=linear_regression(X_train,Y_train)

# @measure_energy
# def test_linear_regression_inference():
#     y_infer=linear_regression_model.predict(X_infer)
#     return y_infer

prediction_list_linear_regression=[]
def linear_regression_prediction(X_train,Y_train,X_test,Y_test):
    print("The below details are for Linear Regression..")
    linear_regression_model=linear_regression(X_train,Y_train)
    predicted_entropy = linear_regression_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    prediction_list_linear_regression.append("test_linear_regression")
    prediction_list_linear_regression.append(r2)
    prediction_list_linear_regression.append(mae)
    prediction_list_linear_regression.append(mse)
    prediction_list_linear_regression.append(rmse)
    return prediction_list_linear_regression

def gaussian_regression(X_train,Y_train):
    gaussian_regression_model = GaussianProcessRegressor()
    return gaussian_regression_model.fit(X_train,Y_train)


@measure_energy(handler=pandas_handler)
def test_gaussian_regression(X_train,Y_train):
    gaussian_regression(X_train,Y_train)
    


# @measure_energy(handler=csv_handler)
# def test_test_gaussian_regression_inference():
#     y_infer=gaussian_regression_model.predict(X_infer)
#     return y_infer


prediction_list_gaussian_regression=[]
def gaussian_regression_prediction(X_train,Y_train,X_test,Y_test):
    print("The below details are for Gaussian Regression..")
    gaussian_regression_model=gaussian_regression(X_train,Y_train)
    predicted_entropy = gaussian_regression_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    prediction_list_gaussian_regression.append("test_gaussian_regression")
    prediction_list_gaussian_regression.append(r2)
    prediction_list_gaussian_regression.append(mae)
    prediction_list_gaussian_regression.append(mse)
    prediction_list_gaussian_regression.append(rmse)
    return prediction_list_gaussian_regression
    


def decision_tree_regression(X_train,Y_train):
    decision_tree_regression_model = DecisionTreeRegressor(random_state = 0) 
    return decision_tree_regression_model.fit(X_train,Y_train)

@measure_energy(handler=pandas_handler)
def test_decision_tree_regression(X_train,Y_train):
    decision_tree_regression(X_train,Y_train)

# decision_tree_regression_model = decision_tree_regression()

# @measure_energy(handler=csv_handler)
# def test_decision_tree_regression_inference():
#     y_infer=decision_tree_regression_model.predict(X_infer)
#     return y_infer

prediction_list_decision_tree_regression=[]
def decision_tree_regression_prediction(X_train,Y_train,X_test,Y_test):
    print("The below details are for Decision tree Regression..")
    decision_tree_regression_model=decision_tree_regression(X_train,Y_train)
    predicted_entropy = decision_tree_regression_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    prediction_list_decision_tree_regression.append("test_decision_tree_regression")
    prediction_list_decision_tree_regression.append(r2)
    prediction_list_decision_tree_regression.append(mae)
    prediction_list_decision_tree_regression.append(mse)
    prediction_list_decision_tree_regression.append(rmse)
    return prediction_list_decision_tree_regression
   




def support_vector_regression(X1,Y1):
    svr_regression_model= SVR()
    return svr_regression_model.fit(X1,Y1.ravel())

@measure_energy(handler=pandas_handler)
def test_support_vector_regression(X1,Y1):
    support_vector_regression(X1,Y1)

# svr_regression_model=support_vector_regression()

# @measure_energy(handler=csv_handler)
# def test_support_vector_regression_inference():
#     y_infer=svr_regression_model.predict(X_infer)
#     return y_infer

prediction_list_support_vector_regression=[]
def support_vector_regression_prediction(X_train,Y_train,X_test,Y_test):
    print("The below details are for support vector Regression..")
    svr_regression_model=support_vector_regression(X_train,Y_train)
    predicted_entropy = svr_regression_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    prediction_list_support_vector_regression.append("test_support_vector_regression")
    prediction_list_support_vector_regression.append(r2)
    prediction_list_support_vector_regression.append(mae)
    prediction_list_support_vector_regression.append(mse)
    prediction_list_support_vector_regression.append(rmse)
    return prediction_list_support_vector_regression


    


def neural_network_regression(X1,Y1):
    neural_networkn_model= MLPRegressor()
    return neural_networkn_model.fit(X1,Y1.ravel())

@measure_energy(handler=pandas_handler)
def test_neural_network_regression(X1,Y1):
    neural_network_regression(X1,Y1)

prediction_list_neural_network_regression=[]
def neural_network_regression_prediction(X_train,Y_train,X_test,Y_test):
    print("The below details are for Neural Network Regression..")
    neural_networkn_model=neural_network_regression(X_train,Y_train)
    predicted_entropy = neural_networkn_model.predict(X_test)
    r2=r2_score(Y_test,predicted_entropy)
    mae=mean_absolute_error(Y_test,predicted_entropy)
    mse=mean_squared_error(Y_test,predicted_entropy)
    rmse=np.sqrt(mse)
    prediction_list_neural_network_regression.append("test_neural_network_regression")
    prediction_list_neural_network_regression.append(r2)
    prediction_list_neural_network_regression.append(mae)
    prediction_list_neural_network_regression.append(mse)
    prediction_list_neural_network_regression.append(rmse)
    return prediction_list_neural_network_regression
    





