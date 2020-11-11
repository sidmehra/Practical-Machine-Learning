# Income-Classification-
Machine Learning, Binary Classification, Scikit-learn 

**Tools Used**

    Scikit-learn, Python 3, NumPy, Pandas, Spyder IDE
    
**Dataset Used**

- [Adult Income Dataset](http://archive.ics.uci.edu/ml/datasets/Adult)

**Project Description and Workflow** 

    1) Balanced the training set using SMOTE and test data is kept side for final model evaluation. 
    2) Evaluate the performance of 8 different Machine learning models using 10-fold stratified cross validation on the balanced training set. 
    3) Based on the Mean Cross Validation Accuracy and Mean Cross Validation F1-score on the training set, we choose the top 3 models for Hyperparameter Tuning. 
    4) We then hyper-parameter tune the top models to get the optimal parameters. 
    5) Compared the performance of the top models on the unbalanced test set before and after hyper-parameter optimization using appropriate evaluation metrics.  
    6) We then incorporate the various Feature selection mechanisms into our Machine learning pipeline. 
    7) We compared the test set performance of the tuned best models before and after Feature Selection.    

**Related Project Files**

    1) Income_prediction.py - Data Preparation - Model Building - Hyper-parameter Optimization - Model
    Evaluation
    
    2) Evaluation_Graphs_Section_4.pdf - Contains the code for plotting of the graphs in the section 4 (Evaluation).
    
    3) Feature_Selection.pdf - We have tuned models (Random Forest and Gradient Boosting) at this stage. 
    This jupyter notebook contains the code for each of Feature Selection mechanisms whose performance is evaluated on the test set using these tuned models. 
    
    4) Solution_Report - This is the report file containing the description and results of this research project.
    
 **Final Results**
 
    Best Result - F1 Score of 0.89 - Tuned Random Forest.
