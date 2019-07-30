import pandas as pd


# importing the wine data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

# Partioning the data and creating the traning and the test data 

from sklearn.model_selection import train_test_split
X,y = df.iloc[:,1:].values, df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
# stratify ensures that both training and test data set have same class proportion

# Scaling the features, min max methods
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# Scaling via standardisation

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)


# Calculating the EIgenvvectors and Eigenvalues

import numpy as np
cov_mat = np.cov(X_train_std.T) # Calculating the covariance matrix
eigen_vals , eigen_vecs = np.linalg.eig(cov_mat)
print('\n Eigenvalus \n%s' %eigen_vals)

""" We want to reduce the dimensionality of the dataset by compressing it onto
    a new feature subspace, we select the subset of eigenvectors that contain 
    the most information. The eigenvectors with large eigenvlaues are selected"""

# The variance explained ratio of the eigenvalues is the eigenvalue lambda(j) and the total sum of 
#  the eigenvalues
# Calculation of cumulative sum of the variances and plotting it
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]

#Cumulative sum of the explained variences. This will help us know what are the most important 
# Eigenvectors
cum_var_exp = np.cumsum(var_exp)

import matplotlib.pyplot as plt
plt.bar(range(1,14), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel("Principal Component Index")
plt.legend('best')
plt.title('Total and Explained Variance')
plt.show()

""" FEATURE TRANSFORMATION """
# Sorting the eigenpairs in decreasing order of the eigenvalues 
eigen_pairs = [np.abs(eigen_vals[i], eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k:k[0], reverse=True)

# We select two eigenvectors that corresponds to the two largest eigenvalues
# to caputure the 60% (from the last graph) of the variance
# w is used for projecting the original data onto the new feature subspace
#np.newaxis is ued to change the dimensions of the already exiting array.

w = np.hstack((eigen_pairs[0][:, np.newaxis].real, eigen_pairs[1][:, np.newaxis].real))
print('Matrix W:\n',w)


X_train_pca = X_train_std.dot(w)

""" Visualizing the transforemd Wine trainig data set, now stored as an 124X2 dimensional matrix
    in a two simensional scatterplot"""

colors = ['r','b','g']
markers = ['s','x','o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == 1,0], X_train_pca[y_train==1,1], c=c, label=1, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend('best')
plt.title('Transformed Wine Training Dataset')
plt.show()