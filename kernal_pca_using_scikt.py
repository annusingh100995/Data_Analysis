from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from sklearn.decomposition import KernelPCA
X , y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_sci_ker_pca = scikit_kpca.fit_transform(X)

plt.scatter(X_sci_ker_pca[y==0,0], X_sci_ker_pca[y==0,1], color='red', marker='^', alpha=0.5)
plt.scatter(X_sci_ker_pca[y==1,0], X_sci_ker_pca[y==1,1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC 1')
plt.xlabel('PC 2')
plt.title('Kernel using Scikit')
plt.show()