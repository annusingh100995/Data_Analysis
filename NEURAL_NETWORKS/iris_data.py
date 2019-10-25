import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)



y = df.iloc[0:100,4].values # from row 0- 100 take the 4th column
y = np.where(y == 'Iris-setosa',-1,1) # where y = Iris setosa set it -1
X = df.iloc[0:100,[0,2]].values

plt.scatter(X[:50,0], X[:50,1],color='red',marker='o',label='setose')
plt.scatter(X[50:100,0], X[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('SEPAL LENGHT(CM)')
plt.ylabel('PETAL LENGHT(CM)')
plt.legend(loc='upper left')
plt.show()