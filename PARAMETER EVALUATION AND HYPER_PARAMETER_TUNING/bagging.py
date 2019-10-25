
""" Bagging 
        Bagging, an ensemble learning techniques, is similar to majority vote classifier.
        In bagging, the bootstrap samples are drawn from initial training set with replacement. """


# importing the wine data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

df_wine.columns = ['Class label','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',
    'Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines',
    'Proline']

df_wine = df_wine[df_wine['Class label'] !=1]
y = df_wine['Class label'].values 
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values  

# Encoding the class labeles into binary format and splitting the data into 80-20% 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

"""Bagging classifier is already implemented in skleanr. 
    Here, I have used unused decision tree as the base classifier. ANd created an ensemble of 500 decision trees on
    different bootstrap samples of the training set"""

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)

bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, bootstrap=True, 
    bootstrap_features=False, n_jobs=1, random_state=1)

from sklearn.metrics import accuracy_score

tree = tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision Tree Train/Test prdict accuracy score %.3f/%.3f' %(tree_train, tree_test))

# Applying bagging

bag = bag.fit(X_train, y_train)
y_train_pred_bag = bag.predict(X_train)
y_test_pred_bag = bag.predict(X_test)
bag_train = accuracy_score(y_train,y_train_pred_bag)
bag_test = accuracy_score(y_test , y_test_pred_bag)
print('Bagging Train/Test predict accuracy score %.3f/%.3f' %(bag_train, bag_test))

#Plotting the graph

x_min = X_train[:,0].min()-1
x_max = X_train[:,0].max()+1
y_min = X_train[:,1].min()-1
y_max = X_train[:,1].max()+1

# preparing the mesh, the two axese
xx, yy= np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# making the subplots
f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8,3))

# for bag and tree classfieris, first fit, then predict,
for idx, clf, tt in zip([0,1], [bag,tree],['Bagging', 'Tree']):
    clf.fit(X_train, y_train)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    axarr[idx].contour(xx, yy, z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0,0], X_train[y_train==0,1], c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='green', marker='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.text(10.2, -1.2, s = 'OD280/OD315 of diluted wine', ha='center', va='center', fontsize=12)
plt.show()