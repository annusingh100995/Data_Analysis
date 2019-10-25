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

# Making the pipeline for logistics regression
# Standarization followed by decreasing the 30 components to 2 components followed by the taining using logistics regression
# make_pipeline is used to create a pipeline of many processes namely, scaling, dimensionality reductiona dn learning algorithmS
 
pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1,solver='lbfgs'))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' %pipe_lr.score(X_test, y_test))

#K fold cross validation to access the model performance

import numpy as np
from sklearn.model_selection import StratifiedKFold

# distibutes the data into 10 sets and each set is used as the test fold and this is iterested for 10 times
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
scores = []

for k , (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' %(k+1, np.bincount(y_train[train]), score))

print('\n CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# scikit learn has a bult in scorer to make the last step less verbose

from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr, X = X_train, y = y_train, cv = 10, n_jobs = 1 )
print('CV accuracy scores : %s' %scores)
print('\n CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Plotting validation curves 

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', random_state=1,solver='lbfgs'))

train_sizes , train_scores , test_scores = learning_curve(estimator = pipe_lr, X = X_train, y = y_train, 
    train_sizes=np.linspace(0.1,1.0,10), cv =10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o',markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean+train_std,train_mean-train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', marker='x',markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean+test_std,test_mean-test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Validation curve')
plt.ylim([0.8,1.0])
plt.show()

