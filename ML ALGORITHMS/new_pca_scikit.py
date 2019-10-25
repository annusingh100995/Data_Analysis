import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier,test_idx=None, resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','pink','lightgreen','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min ,x1_max = X[:,0].min()-1, X[:,0].max() + 1
    x2_min ,x2_max = X[:,0].min()-1, X[:,0].max() + 1
    # creating the meshgrid, starting from x1_min to x1_max with increment of resolution
    xx1 , xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx , c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y==c1,0], y=X[y == c1,1], alpha=0.8, c=colors[idx], marker=markers[idx], label=c1, edgecolor = 'black')
    if test_idx:
        X_test , y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1] , c='',edgecolor='black', alpha=1.0,linewidth=1, marker='o', s=100,label='test_set')


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
lr = LogisticRegression(solver='lbfgs',multi_class='auto')
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca,y_train,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='best')
plt.title('PCA ANALYSIS')
plt.show()

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='best')
plt.title('LDA ANALYSIS')
plt.show()
