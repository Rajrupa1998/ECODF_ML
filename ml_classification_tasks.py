import pandas as pd
import numpy as np
import time
from pyJoules.handler.pandas_handler import PandasHandler
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from  sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LinearRegression
import random


pandas_handler = PandasHandler()


def store_values():
    df = pandas_handler.get_dataframe()
    return df



# Y1=Y1.astype(int)

def random_forest_classification(X1,Y1):
    random_forest_model = RandomForestClassifier()
    return random_forest_model.fit(X1,Y1.ravel())

# random_forest_model=random_forest_classification()

# @measure_energy(handler=csv_handler)
# def test_random_forest_classification_inference():
#     y_infer=random_forest_model.predict(X_infer)
#     return y_infer

@measure_energy(handler=pandas_handler)
def test_random_forest_classification(X1,Y1):
    random_forest_classification(X1,Y1)

prediction_list_random_forest_classification=[]
def random_forest_classification_prediction(X_train,Y_train,X_test,Y_test):
    print("The below details are for Random forest classification..")
    random_forest_model=random_forest_classification(X_train,Y_train)
    score=random_forest_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = random_forest_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
   #print("Accuracy= ",(tp+tn)/(tp+tn+fp+fn))
    accuracy= (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    specificity=tn/tn+fp
    f1score= 2 *(recall*precision)/(precision+recall)
    prediction_list_random_forest_classification.append("test_random_forest_classification")
    prediction_list_random_forest_classification.append(score)
    prediction_list_random_forest_classification.append(accuracy)
    prediction_list_random_forest_classification.append(recall)
    prediction_list_random_forest_classification.append(precision)
    prediction_list_random_forest_classification.append(specificity)
    prediction_list_random_forest_classification.append(f1score)
    return prediction_list_random_forest_classification
    # print("f1 score= ", f1score) 



def logistic_regressor_classification(X1,Y1):
    logistic_regression_model = LogisticRegression()
    return logistic_regression_model.fit(X1,Y1.ravel())



@measure_energy(handler=pandas_handler)
def test_logistic_regressor_classification(X1,Y1):
    logistic_regressor_classification(X1,Y1)

# logistic_regression_model=logistic_regression_classification()

# @measure_energy(handler=csv_handler)
# def test_logistic_regression_classification_inference():
#     y_infer=logistic_regression_model.predict(X_infer)
#     return y_infer

prediction_list_logistic_regression_classification=[]
def logistic_regressor_classification_prediction(X_train,Y_train,X_test,Y_test):
    print("The below details are for Logistic Regression classification..")
    logistic_regression_model=logistic_regressor_classification(X_train,Y_train)
    score=logistic_regression_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = logistic_regression_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
    accuracy= (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    specificity=tn/tn+fp
    f1score= 2 *(recall*precision)/(precision+recall)
    prediction_list_logistic_regression_classification.append("test_logistic_regressor_classification")
    prediction_list_logistic_regression_classification.append(score)
    prediction_list_logistic_regression_classification.append(accuracy)
    prediction_list_logistic_regression_classification.append(recall)
    prediction_list_logistic_regression_classification.append(precision)
    prediction_list_logistic_regression_classification.append(specificity)
    prediction_list_logistic_regression_classification.append(f1score)
    return prediction_list_logistic_regression_classification
    # print("f1 score= ", f1score) 


def gaussian_NB_classification(X1,Y1):
    naive_bayes_model = GaussianNB()
    return naive_bayes_model.fit(X1,Y1.ravel())

@measure_energy(handler=pandas_handler)
def test_gaussian_NB_classification(X1,Y1):
    gaussian_NB_classification(X1,Y1)

# naive_bayes_model=gaussian_NB_classification()

# @measure_energy(handler=csv_handler)
# def test_gaussian_NB_classification_inference():
#     y_infer=naive_bayes_model.predict(X_infer)
#     return y_infer

prediction_list_gaussian_NB_classification=[]
def gaussian_NB_classification_prediction(X_train,Y_train,X_test,Y_test):  
    print("The below details are for Naive Bayes classification..")
    naive_bayes_model=gaussian_NB_classification(X_train,Y_train)
    score=naive_bayes_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = naive_bayes_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
    accuracy= (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    specificity=tn/tn+fp
    f1score= 2 *(recall*precision)/(precision+recall)
    prediction_list_gaussian_NB_classification.append("test_gaussian_NB_classification")
    prediction_list_gaussian_NB_classification.append(score)
    prediction_list_gaussian_NB_classification.append(accuracy)
    prediction_list_gaussian_NB_classification.append(recall)
    prediction_list_gaussian_NB_classification.append(precision)
    prediction_list_gaussian_NB_classification.append(specificity)
    prediction_list_gaussian_NB_classification.append(f1score)
    return prediction_list_gaussian_NB_classification



def SVM_classification(X1,Y1):
    svm_classifier_model = SVC()
    return svm_classifier_model.fit(X1,Y1.ravel())

@measure_energy(handler=pandas_handler)
def test_SVM_classification(X1,Y1):
    SVM_classification(X1,Y1)


# svm_classifier_model=SVM_classification()

# @measure_energy(handler=csv_handler)
# def test_SVM_classification_inference():
#     y_infer=svm_classifier_model.predict(X_infer)
#     return y_infer

prediction_list_svm_classification=[]
def SVM_classification_prediction(X_train,Y_train,X_test,Y_test):  
    print("The below details are for SVM classification..")
    svm_classifier_model=SVM_classification(X_train,Y_train)
    score=svm_classifier_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = svm_classifier_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
    accuracy= (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    specificity=tn/tn+fp
    f1score= 2 *(recall*precision)/(precision+recall)
    prediction_list_svm_classification.append("test_SVM_classification")
    prediction_list_svm_classification.append(score)
    prediction_list_svm_classification.append(accuracy)
    prediction_list_svm_classification.append(recall)
    prediction_list_svm_classification.append(precision)
    prediction_list_svm_classification.append(specificity)
    prediction_list_svm_classification.append(f1score)
    return prediction_list_svm_classification





def decision_tree_classification(X1,Y1):
    decision_tree_classifier_model = SVC()
    return decision_tree_classifier_model.fit(X1,Y1.ravel())

@measure_energy(handler=pandas_handler)
def test_decision_tree_classification(X1,Y1):
    decision_tree_classification(X1,Y1)



# decision_tree_classifier_model=decision_tree_classification()
# @measure_energy(handler=csv_handler)
# def test_decision_tree_classification_inference():
#     y_infer=decision_tree_classifier_model.predict(X_infer)
#     return y_infer

prediction_list_decision_tree_classification=[]
def decision_tree_cassification_prediction(X_train,Y_train,X_test,Y_test):  
    print("The below details are for Decision tree classification..")
    decision_tree_classifier_model=decision_tree_classification(X_train,Y_train)
    score=decision_tree_classifier_model.score(X_test,Y_test)
    print("Model score= ",score)
    predicted_entropy = decision_tree_classifier_model.predict(X_test)
    cm = confusion_matrix(Y_test,predicted_entropy)
    tn, fp, fn, tp = cm.ravel()
    accuracy= (tp+tn)/(tp+tn+fp+fn)
    recall = tp/(tp+fn)
    precision=tp/(tp+fp)
    specificity=tn/tn+fp
    f1score= 2 *(recall*precision)/(precision+recall)
    prediction_list_decision_tree_classification.append("test_decision_tree_classification")
    prediction_list_decision_tree_classification.append(score)
    prediction_list_decision_tree_classification.append(accuracy)
    prediction_list_decision_tree_classification.append(recall)
    prediction_list_decision_tree_classification.append(precision)
    prediction_list_decision_tree_classification.append(specificity)
    prediction_list_decision_tree_classification.append(f1score)
    return prediction_list_decision_tree_classification   





