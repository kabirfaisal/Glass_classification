# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 22:17:04 2022

@author: MdFKabir
"""
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
  
# load dataset and print basic information
def load_data():
    df = pd.read_csv(r"C:\Users\Kabir\Downloads\Glass Classification\glass.csv",sep= ',', header = None)
      
    # Print the dataset length, shape, data type
    print ("Length: ", len(df))
    print ("Shape: ", df.shape)
    print ("Shape: ", df.dtypes)
      
    # Print the obseravtions
    print (df.head())
    return df
  
# split the dataset as input and output variable
def splitdataset(df):
  
    # Separate input X and target Y 
    X = df.values[:, 0:8]
    Y = df.values[:, 9]
    print ("input Shape: ", X.shape)
    print ("output Shape: ", Y.shape)
  
    # Splitting into train and test set
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
      
    return X, Y, X_train, X_test, y_train, y_test
      
# training with gini
def train_gini(X_train, X_test, y_train):
  
    # Creating the classifier with DecisionTreeClassifier for gini
    gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,  min_samples_leaf=5)
  
    # fit model
    gini.fit(X_train, y_train)
    return gini
      
# training with entropy.
def tarin_entropy(X_train, X_test, y_train):
  
    # Creating the classifier with DecisionTreeClassifier for entropy
    entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, min_samples_leaf=3)
  
    # fit model
    entropy.fit(X_train, y_train)
    return entropy
  
  
# make predictions
def prediction(X_test, pre_value):
  
    # Predicton on test with giniIndex
    y_pred = pre_value.predict(X_test)
    #print("Predicted values:")
    #print(y_pred)
    return y_pred
      
# Function to calculate accuracy
def calculate_acc(y_test, y_pred):
      
    #print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",classification_report(y_test, y_pred))
  
# Driver code
def main():
      
    # Building Phase
    glass_data = load_data()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(glass_data)
    gini = train_gini(X_train, X_test, y_train)
    entropy = tarin_entropy(X_train, X_test, y_train)
      
    # Operational Phase
    print("Results Using Gini Index:")
      
    # Prediction using gini
    y_pred_gini = prediction(X_test, gini)
    calculate_acc(y_test, y_pred_gini)
      
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, entropy)
    calculate_acc(y_test, y_pred_entropy)
      
      
# Calling main function
if __name__=="__main__":
    main()