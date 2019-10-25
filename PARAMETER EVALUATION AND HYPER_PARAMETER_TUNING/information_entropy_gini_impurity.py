import matplotlib.pyplot as plt
import numpy as np
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

def gini(p):
    return (p)*(1-(p))+(1-p)*(1-(1-p))

def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2((1-p))

def error(p):
    return 1-np.max([p,1-p])

x = np.arange(0.0,1.0,0.01)

ent = [entropy(p) if p!= 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in x]
err = [error(i) for i in x]
fig  = plt.figure()
ax = plt.subplot(111)
for i , lab , ls, c in zip([ent,sc_ent, gini(x) , err],['Entropy', 'Entropy(scaled)','Gini Impurity','Misclassification Error'],
    ['-','-','--','-.'], ['black','blue','red','green','pink']): line = ax.plot(x,i,label=lab,linestyle=ls,lw=2,color=c)

ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15), ncol=5, fancybox=True,shadow=False)
ax.axhline(y=0.5, linewidth=1,color='k',linestyle='--')
ax.axhline(y=0.5, linewidth=1,color='k',linestyle='--')
plt.ylim([0,1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:,[2,3]] # all rows and column 2 and 3
y = iris.target
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y,test_size= 0.3, random_state=1, stratify=y)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(X_train , y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,classifier=tree, test_idx=range(105,150))
plt.xlabel=('petal length')
plt.ylabel=('petal width')
plt.legend(loc='upper left')
plt.show()

from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree, filled=True, rounded=True,class_names=['Setosa','Versicolor','Virginia'],
    feature_names =['petal length','petal width'],
    out_file = None)

graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')