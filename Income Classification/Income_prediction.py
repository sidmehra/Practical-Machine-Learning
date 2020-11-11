# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 14:06:50 2019

@author: sid
"""

# Importing necessary packages with their alias 
import numpy as np # For Numerical computations 
import pandas as pd # For Data analysis and Manipulation
import matplotlib.pyplot as plt # For Data Visualization
import seaborn as sns # For more better visualization 
from warnings import simplefilter # import warnings filter
simplefilter(action='ignore', category=FutureWarning) # ignore all future warnings
# To load the dataset into a pandas data-frame 
adult_DF = pd.read_csv("adult.csv") # Now adult_DF is a data-frame 
# How many instances and number of features 
#print("Shape of the raw dataframe is:",adult_DF.shape)

# ---------------------DATA PREPARATION / PRE-PROCESSING PHASE --------------------------------------

# STEP 1 ------------> HANDLING THE OUTLIERS (optional)
# Selecting the continuous features 
# Univariate outlier detection of the continuous features which are age, fnlwgt, capital-gain, capital-loss, hours-per-week
# Box plot of age feature 
sns.boxplot(adult_DF.iloc[:, 0])
plt.show()
# Box plot of fnlwgt feature 
sns.boxplot(adult_DF.iloc[:, 2])
plt.show()
# Box plot of capital-gain feature 
sns.boxplot(adult_DF.iloc[:, 10])
plt.show()
# Box plot of capital-loss feature 
sns.boxplot(adult_DF.iloc[:, 11])
plt.show()
# Box plot of hours-per-week feature 
sns.boxplot(adult_DF.iloc[:, 12])
plt.show()
 
# STEP 2 -----------> HANDLING MISSING VALUES 
# Replacing the "?" with "NaN" to count the occurances of missing values accross each feature
adult_DF = adult_DF.replace('?', np.NaN)
# How many missing values are there accross each feature ? 
#print(adult_DF.isnull().sum())
# As we are only having 5% of rows with missing data, I choose to delete those rows
adult_DF= adult_DF.dropna()
# Is there any missing values left in the dataframe ?  
#print(adult_DF.isnull().values.any())

# STEP 3 -----------> FEATURE ENCODING
# Encoding the categorical features (both ordinal and nominal)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
adult_DF = adult_DF.apply(encoder.fit_transform)
# We have the exact same numerical representation of education and educational-num feature which depicts the same ordering.
# Dropping the educational-num column from the dataset 
adult_DF= adult_DF.drop(['educational-num'], axis=1)
# Shape after encoding the categorical features 
#print("Shape after encoding the categorical features is:", adult_DF.shape)

# Firstly we will divide our dataframe into predictor variables and target variable (income)
predictorDF = adult_DF.iloc[:, :-1].values
targetDF = adult_DF.iloc[:, 13].values

# STEP 4 -----------> FEATURE SCALING
# Now we will perform the standard scaling only to predictor variables not to the target variable
# Creating the object of StandardScaler class for standardizing all the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
predictorDF = scaler.fit_transform(predictorDF)

# Dividing our dataset into train and test set 
from sklearn.model_selection import train_test_split
predictorDF_train, predictorDF_test, targetDF_train, targetDF_test = train_test_split(predictorDF, targetDF, test_size = 0.2, random_state = 0, stratify=targetDF)

# STEP 5 -----------> HANDLING THE IMBALANCE   
# Only the training data needs to be balanced, not the entire dataset
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
# Oversampling the training data instances 
predictorDF_train, targetDF_train = sm.fit_sample(predictorDF_train, targetDF_train)

# STEP 6 -------------> FEATURE SELECTION 
# This would be my research component that would be integrated into my code only after getting the initial results as I need to see its impact on the overall results.  
# REFER THE JUPYTER NOTEBOOK - Feature_Selection.ipynb

# ---------------------BUILD RANGE OF ML MODELS AND EVALUATE THEIR PERFORMANCE  --------------------------------------

# ----------------------------------CODE FOR SECTION 3.2 a)----------------------------------------------------

# Build range of machine learning models with default Hyperparameters
# Building the LOGISTIC REGRESSION model 
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state=0)
# Building the KNN model 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
# Building the SVM model 
from sklearn.svm import SVC
svm = SVC(random_state=0)
# Building the NAIVE BAYES model 
from sklearn.naive_bayes import GaussianNB
bayes = GaussianNB()
# Building the DECISION TREE model
from sklearn.tree import DecisionTreeClassifier
decisionTree = DecisionTreeClassifier(random_state=0)
# Building the RANDOM FOREST model 
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier(random_state=0)
# Building the GRADIENT BOOSTING CLASSIFIER 
from sklearn.ensemble import GradientBoostingClassifier
gradientBoosting = GradientBoostingClassifier(random_state=0)
# Building the ADA BOOST CLASSIFIER 
from sklearn.ensemble import AdaBoostClassifier
adaBoost = AdaBoostClassifier(random_state=0)

'''
# ----------------------------------CODE FOR SECTION 4.1 a) and  4.2 a)----------------------------------------------------

# TOP 3 MODELS THAT WILL UNDERGO HYPER-PARAMETER OPTIMIZATION 
# Would be using 10-fold stratified cross validation for evaluating the model performance. 
# Calculate the performance of 8 different models on the balanced training set 

model_names = ['Logistic Regression', 'KNN', 'SVM', 'Naive Bayes', 'Decision tree', 'Random forest', 'Gradient Boosting', 'Ada Boost']
models = [logistic, knn, svm, bayes, decisionTree, randomForest, gradientBoosting, adaBoost]
from sklearn.model_selection import cross_val_score
# Evaluating the performance of the different models 
for model_name, model in zip(model_names, models):
     print("Model name:{}".format(model_name))
     for score in ["accuracy", "precision", "recall", "f1"]:
         # Applying stratified k-fold cross validation on balanced training set  
         scores = cross_val_score(model, predictorDF_train, targetDF_train, cv=10, scoring=score)
         np_scores=np.array(scores)
         # Computing the mean of the all the accuracies 
         mean_score = np.mean(np_scores)
         print("The mean {} score is: {} ".format(score, mean_score))
     print("\n")

# Top 3 most promising models that are selected for hyper-parameter optimization are based on the best cv mean accuracy and mean f1 score 
# 1) GRADIENT BOOSTING (Mean Accuracy=86.43, Mean F1_score=86.51)
# 2) ADA BOOST (Mean Accuracy=85.16, Mean F1_score= 85.29)
# 3) RANDOM FOREST (Mean Accuracy=87.31, Mean F1_score= 86.72) 

'''  

'''
# ----------------------------------CODE FOR SECTION 4.1 b) and 4.2 b)----------------------------------------------------

# -------------------EVALUATE THE PERFORMANCE OF TOP MODELS WITH DEFAULT PARAMETERS ON THE TEST SET  --------------------------------------

# Evaluate the performance on the test set using different EVALUATION METRICS  

# Evaluation from default RANDOM FOREST model and GRADIENT BOOSTING model 

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# These are the default classifiers with optimal hyper-parameters for both the models 
defaultRF = RandomForestClassifier(random_state=0)
defaultGB = GradientBoostingClassifier(random_state=0)

model_names = ["Random Forest", "Gradient Boosting"]
default_models = [defaultRF, defaultGB]

# Evaluating the Model Performance on the test set based on different evaluation metrics 
# 1) ACCURACY 
# 2) PRECISION 
# 3) RECALL
# 4) F1-SCORE
# 5) CONFUSION MATRIX 

for model_name, default_model in zip(model_names, default_models):
    default_model = default_model.fit(predictorDF_train, targetDF_train)
    results= default_model.predict(predictorDF_test)
    accuracy = accuracy_score(results, targetDF_test)
    print("TEST ACCURACY on default {} is:{} ".format(model_name, accuracy))
    f1Score = f1_score(results, targetDF_test)
    print("F1 SCORE on default {} is:{} ".format(model_name, f1Score))
    report = classification_report(targetDF_test, results)
    print("The CLASSIFICATION REPORT for default {} is: {}\n".format(model_name, report))
    print("The CONFUSION MATRIX of {} is:".format(model_name))
    print(confusion_matrix(targetDF_test,results))
    print("----------------------------------------------------------------------------------------")
    print("\n")

# FINAL RESULTS ON THE TEST SET FOR DIFFERENT EVALUATION METRICS ON DEFAULT RANDOM FOREST AND GRADIENT BOOSTING  

#----------------------------------------------------------------------------------------------
    
# RANDOM FOREST    
# Accuracy ----------------------> 83.17  %
# F1-Score ------------------------> 65.82   % 
# Results from the Confusion Matrix of default RANDOM FOREST 
# 1) 89.03 % of the majority income group (<=50k income range) is predicted correctly 
# 2) 65.00 % of the minority income group (>50k income range) is predicted correctly 

# ----------------------------------------------------------------------------------------------

# GRADIENT BOOSTING   
# Accuracy ----------------------> 82.70 % 
# F1-Score ------------------------> 69.00  %     
# Results from the Confusion Matrix of the default GRADIENT BOOSTING  
# 1) 84.37 % of the majority income group (<=50k income range) is predicted correctly
# 2) 77.65 % of the minority income group (>50k income range) is predicted correctly 
    
'''
  
'''
# ----------------------------------CODE FOR SECTION 4.2 a)----------------------------------------------------

# ---------------------HYPER-PARAMETER OPTIMIZATION OF TOP MODELS  --------------------------------------

# Would be choosing the RANDOM FOREST and GRADIENT BOOSTING for hyper-parameter tuning 

from sklearn.model_selection import RandomizedSearchCV
# Tuning the RANDOM FOREST Model 
parameters = { 
    'n_estimators': [10, 100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'bootstrap': [True,False],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
}
rand = RandomizedSearchCV(randomForest, param_distributions=parameters, cv=10, scoring='accuracy', n_iter=50, random_state=5)
rand.fit(predictorDF_train, targetDF_train)
print("The best parameters are:\n", rand.best_params_)
print("The best score is:\n", rand.best_score_)

# After Hyperparameter Tuning the Random Forest, we get the following best hyper-parameters 
#------------------------------Best PARAMETERS of RANDOM FOREST-----------------------------------------
#                                   bootstrap= True
#                                   max_depth=70
#                                   max_features= 'auto'
#                                   min_samples_leaf= 4
#                                   min_samples_split= 10
#                                   n_estimators= 400 

#-------------------------------Best PERFORMANCE on Training Set------------------------------------  
#                                   Best Mean Accuracy - 87.83
#                                   Best Mean F1 Score - 89.65

# Tuning the GRADIENT BOOSTING Model  
parameters = {
    'learning_rate':[0.15,0.1,0.05,0.01],
    'min_samples_split':[20,40,60,100],
    'min_samples_leaf':[1,3,5,7,9],
    'max_depth':[3,4,5,6],
    'max_features':[2,3,4,5,6,7],
    "subsample":[0.8, 0.85, 0.9, 0.95, 1.0],
    'n_estimators':[750,1000,1500,1750]
    }
rand = RandomizedSearchCV(gradientBoosting, param_distributions=parameters, cv=10, scoring='accuracy', n_iter=50, random_state=5)
rand.fit(predictorDF_train, targetDF_train)
print("The best parameters are:\n", rand.best_params_)
print("The best score is:\n", rand.best_score_)

# After Hyperparameter Tuning the Gradient Boosting, we get the following best hyper-parameters 
#-------------------------------Best Parameters of GRADIENT BOOSTING------------------------------------------ 
#                                          learning_rate=0.01 
#                                          n_estimators=1500
#                                          max_depth=4 
#                                          min_samples_split=40
#                                          min_samples_leaf=7
#                                          max_features=4 
#                                          subsample=0.95
#----------------------------------Best Performance on Training Set----------------------------------------------  
#                                       Best Mean Accuracy - 85.21
#                                       Best Mean F1 Score - 86.94

'''

# ----------------------------------CODE FOR SECTION 4.2 b)----------------------------------------------------

# -------------------EVALUATE THE PERFORMANCE OF TOP MODELS WITH TUNED PARAMETERS ON THE TEST SET  --------------------------------------

# Evaluate the performance on the test set using different EVALUATION METRICS  

# Evaluation from tuned RANDOM FOREST model and GRADIENT BOOSTING model 

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# These are the tuned classifiers with optimal hyper-parameters for both the models 
tunedRF = RandomForestClassifier(bootstrap= True, max_depth=70, max_features= 'auto', min_samples_leaf= 4, min_samples_split= 10, n_estimators= 400, random_state=0)
tunedGB = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1500, max_depth=4, min_samples_split=40, min_samples_leaf=7, max_features=4,subsample=0.95, random_state=0)

model_names = ["Random Forest", "Gradient Boosting"]
tuned_models = [tunedRF, tunedGB]

# Evaluating the Model Performance on the test set based on different evaluation metrics 
# 1) ACCURACY 
# 2) PRECISION 
# 3) RECALL
# 4) F1-SCORE
# 5) CONFUSION MATRIX 

for model_name, tuned_model in zip(model_names, tuned_models):
    tuned_model = tuned_model.fit(predictorDF_train, targetDF_train)
    results= tuned_model.predict(predictorDF_test)
    accuracy = accuracy_score(results, targetDF_test)
    print("TEST ACCURACY on tuned {} is:{} ".format(model_name, accuracy))
    f1Score = f1_score(results, targetDF_test)
    print("F1 SCORE on tuned {} is:{} ".format(model_name, f1Score))
    report = classification_report(targetDF_test, results)
    print("The CLASSIFICATION REPORT for tuned {} is: {}\n".format(model_name, report))
    print("The CONFUSION MATRIX of {} is:".format(model_name))
    print(confusion_matrix(targetDF_test,results))
    print("----------------------------------------------------------------------------------------")
    print("\n")

# FINAL RESULTS ON THE TEST SET FOR DIFFERENT EVALUATION METRICS ON TUNED RANDOM FOREST AND GRADIENT BOOSTING  

#----------------------------------------------------------------------------------------------
    
# RANDOM FOREST    
# Accuracy ----------------------> 83.67 %
# F1-Score ------------------------> 68.75  % 
# Results from the Confusion Matrix of tuned RANDOM FOREST 
# 1) 87.35 % of the majority income group (<=50k income range) is predicted correctly 
# 2) 72.47 % of the minority income group (>50k income range) is predicted correctly 

# ----------------------------------------------------------------------------------------------

# GRADIENT BOOSTING   
# Accuracy ----------------------> 83.60 % 
# F1-Score ------------------------> 69.64 %     
# Results from the Confusion Matrix of the tuned GRADIENT BOOSTING  
# 1) 86.15 % of the majority income group (<=50k income range) is predicted correctly
# 2) 75.86 % of the minority income group (>50k income range) is predicted correctly 
    
