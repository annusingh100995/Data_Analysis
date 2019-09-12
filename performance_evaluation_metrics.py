
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


from sklearn.metrics import confusion_matrix


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import make_pipeline

#importing the gris search funtion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# preparing the pipeline for grid search, including the scaling and the support vector machine
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

# training the data using support vector machines
pipe_svc.fit(X_train, y_train)

# Predicting the labels for the test data
y_pred = pipe_svc.predict(X_test)

# preparing the condfusion matrix
confmat = confusion_matrix(y_true = y_test, y_pred = y_pred)
print(confmat)

import matplotlib.pyplot as plt

# specifying the size of the figure
fig , ax = plt.subplots(figsize=(2.5,2.5))

# the data, the color 
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

# defining the text position for labelling
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# other scoring metrics are precision and recall 

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

print('Precision: %.3f' % precision_score(y_true=y_test, y_pred= y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1 Score:%.3f' % f1_score(y_true=y_test, y_pred=y_pred))


#importing the gris search funtion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# preparing the pipeline for grid search, including the scaling and the support vector machine
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C':param_range, 'svc__kernel':['linear']}, 
    {'svc__C':param_range, 'svc__gamma':param_range, 'svc__kernel':['rbf']}]

# We can change the scoring method in the grid search by the following method

from sklearn.metrics import make_scorer, f1_score
scorer = make_scorer(f1_score, pos_label=0)

# Specifying the parameters,the estimator and the scoring method
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring=scorer, cv=10)

gs = gs.fit(X_train, y_train)

print(gs.best_estimator_)
print(gs.best_score_)
print(gs.best_params_)

# Plotting the Receiver Operating Characteristic

""" ROC is used to select the model for classification based on their performance
    with repect to the True positive rate and the false  positive rate
    The diagonal in a ROC can be interpreted as random guessing, the models below the diagonal are far worse than random guessing
    A perfect classifier would fall into the top left ocrner of the graph with TPR = 1 and FPR = 0 """


from sklearn.metrics import roc_curve, auc
from scipy import interp

import numpy as np
from sklearn.model_selection import StratifiedKFold

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(penalty='l2', random_state=1,C=100.0,solver='lbfgs'))
X_train2 = X_train[:,[4, 14]]

"""  StratifiedKFold is used to divide the data(X_train and y_train) into 3 segments. This then follows the KFold method.
    Then the divided segments are stored in a list"""

cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train, y_train))

# Intialising a figure
fig = plt.figure(figsize=(7, 5))


mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
all_tpr = []

""" For each iteration in the KFold method here, 
    the data is trained using the pipe_lr pipeline, that uses Logistic Regression to train the model.
    Then the test data is used to test and to find the probabilty (how truly/falsiliy it is classified).
    The roc_curve function is used to find the fpr, tpr and the threshold.
    
        The mean is calculated and then tha variable are intilised to 0 for the next iteration"""

for i , (train,test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])
    
    fpr , tpr, thresholds = roc_curve(y_train[test], probas[:,1], pos_label=1)
    mean_tpr += interp(mean_tpr,fpr, tpr)
    mean_tpr = 0.0
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr, tpr, label = 'ROC fold %d (area = %0.2f)' %(i+1, roc_auc))

# Plotting the line for the random guessing
plt.plot([0,1], [0,1], linestyle='--', color=(0.6,0.6,0.6), label='random_guessing')

mean_tpr = mean_fpr/len(mean_fpr)
mean_auc = auc(mean_tpr, mean_fpr)


plt.plot(mean_fpr, mean_tpr, 'k--', label='mean ROC (area = %0.2f)' % mean_auc, lw =2)
plt.plot([0,0,1], [0,1,1], linestyle=':', color='black', label = 'perfect peformance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive')
plt.ylabel('true positive rate')
plt.legend(loc='lower right')
plt.show()