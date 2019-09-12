""" # Grid search is a brute force exhaustive search paradigm where a list of values is specified. And the machine evaluates the model performance
for each combination of those to obtain a optimal combinationof values

"""

# importing libraries

import pandas as pd

# importing the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

# there are 30 fratures and one column for malignant or benign tumor  assign the 30 features to x and the labels to y

X = df.loc[:,2:].values
y = df.loc[:,1].values

# Tranform the B and M labels of the data into numeric data labels 0 and 1 

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# now y is tranformed into numeric labels , malignant = 1 and benign = 0

# Splliting the test and the training data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, stratify=y, random_state=1)

# Combining the transformor and the estimators in a piprline

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import make_pipeline

#importing the gris search funtion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# preparing the pipeline for grid search, including the scaling and the support vector machine
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

""" The grid search will have the following exhaustive searhes ,
        First: with the kernel set to linear and the values of C(inverse of gamma) ranging from the values in param_range
        Second: with the kernel is radial basis function and the values of gamma and C are from the values in param_range """


param_grid = [{'svc__C':param_range, 'svc__kernel':['linear']}, 
    {'svc__C':param_range, 'svc__gamma':param_range, 'svc__kernel':['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs= -1)

#Traning the model
gs = gs.fit(X_train, y_train)

#Printing the best score and the best parameters
print(gs.best_score_)
print(gs.best_params_)