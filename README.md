# Diabetes Prediction for Women

## Problem Statement:

Early detection and prediction of diabetes can help mitigate its progression and reduce associated
health risks, By using this real records, we will try to build a machine learning model to predict
whether or not the WOMEN patient is in the dataset have diabetes or not

## Objectives

- Develop a classification model to predict diabetes in women.
- Clean and preprocess data to enhance model accuracy.
- Visualize results and evaluate model performance.

### Columns:

    Pregnancies : Number of times pregnant 
    
    Glucose : Plasma glucose concentration a 2 hours in an oral glucose tole
    
    BloodPressure : Diastolic blood pressure (mm Hg)
    
    SkinThikness : the measurement of how thick or thin the skin is
    
    Insulin : 2-Hour serum insulin ( mu U/ml)
    
    BMI : Body mass index ( weight in kg / pow(height in m, 2)
    
    DiabetesPedigreeFunction : Diabetes Pedigree Function
    
    Age : in (years)
    
    Outcome : (0 : True : she is   OR 1 : False : she is not)   

## Data

The dataset used is a CSV file named `diabetes.csv` and contains the following columns:

- `Pregnancies` : Number of pregnancies
- `Glucose` : Plasma glucose concentration after 2 hours of an oral glucose tolerance test
- `BloodPressure` : Diastolic blood pressure (mm Hg)
- `SkinThickness` : Skinfold thickness
- `Insulin` : Serum insulin (mu U/ml)
- `BMI` : Body mass index
- `DiabetesPedigreeFunction` : Diabetes pedigree function
- `Age` : Age (in years)
- `Outcome` : Presence or absence of diabetes (0: no, 1: yes)

## Prerequisites

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- mlxtend
- missingno

##  Overview of each of the imports :

- Pandas: for Loading, cleaning, transforming, and analyzing structured data (e.g., CSV files).

- Numpy: Working with multi-dimensional arrays and performing mathematical operations.Pyplot is the most used module.

- Matplotlib: Plotting graphs such as line charts, bar charts, scatter plots, and histograms.

- Seaborn:  Visualizing data distributions, correlations, and relationships through heatmaps, pair plots, etc.

- from mlxtend.plotting import plot_decision_regions: Visualizing how a classification algorithm has separated different classes in the data.

- from pandas.plotting import scatter_matrix:Understanding correlations and patterns between features in the dataset.

- missingno: Checking for and visualizing the extent and patterns of missing data in a dataset.

- from sklearn.preprocessing import StandardScaler: Normalizing data to ensure that features have similar scales, which is important for machine learning models like KNN.

- from sklearn.model_selection import train_test_split: Dividing data into subsets for training and testing machine learning models, ensuring unbiased evaluation.

- from sklearn.neighbors import KNeighborsClassifier: Implementing the K-Nearest Neighbors (KNN) classification algorithm.

- from sklearn import metrics: Calculating evaluation metrics such as accuracy, precision, recall, and F1-score.

- from sklearn.metrics import accuracy_score: Evaluating the proportion of correct predictions made by a classifier.

- from sklearn.metrics import classification_report: Assessing the performance of a classification model with more detail than simple accuracy.


- from sklearn.metrics import roc_curve: Evaluating the performance of a binary classifier, especially in the case of imbalanced data.

- from sklearn.metrics import roc_auc_score: Measuring the effectiveness of a binary classifier. A higher AUC indicates better performance.

- from sklearn.model_selection import GridSearchCV: Hyperparameter tuning to find the optimal set of parameters for a machine learning model.
