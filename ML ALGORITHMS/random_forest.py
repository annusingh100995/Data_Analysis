from sklearn import datasets
import numpy as np


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

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


iris = datasets.load_iris()
X = iris.data[:,[2,3]] # all rows and column 2 and 3
y = iris.target

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3, random_state=1, stratify=y)
# function divides the data into training and test data, test = 30% data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # fit calculates the mean and standard deviation for each feature dimension
X_train_std = sc.transform(X_train) # tansform , transforms the data
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini', n_estimators =25, random_state=1, n_jobs = 2)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.title('Ranodm Forest Algorithm')
plt.show()


"""K-nearest neightbour algorithm """

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p = 2, metric = 'minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined, y_combined, classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.title('KNN Algorithm')
plt.show()
