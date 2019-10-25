from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np 
import matplotlib.pyplot as plt

def rbf_kernel_pca(X, gamma, n_components):
    """ RRBF Kernal PCA Implementation.

    Parameters
    X : numpy array, shape [n_samples,n_features]

    gamma : float
        tuning parameter of the RBF kernel

    n_components : int
        number of principal components to return 
    
    Returns: 
    X_pc : numpy array , shape[n_samples, k_features]
    projected dataset
    """
    # Calculate the pairwise squared euclidean distances in the M*N dimensional dataset
    sq_dist = pdist(X, 'sqeuclidean')

    # Convert the pairwise distance into sqaure matrix
    mat_sq_dist = squareform(sq_dist)
    K = exp(-gamma*mat_sq_dist)
    # Compute the kernal matrix
    N = K.shape[0]
    one_n = np.ones((N,N))/N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernal matrix
    # Scipy.linalg.eigh returns them in ascendig order

    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]

    # Collecting the top k eigenvectors 

    X_pc = np.column_stack((eigvecs[:,i] for i in range(n_components)))

    return X_pc

# MAking the half two moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='^', alpha= 0.5)
plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha= 0.5)
plt.show()

# Fitting PCA to the half moon data set adnd plotting the PCA breakdown of the features
# Plotting the most important principal components of the data

from sklearn.decomposition import PCA
scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)
fig , ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0],X_spca[y == 0, 1], color='red', marker = '^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0],X_spca[y == 1, 1], color='blue', marker = 'o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
plt.title('Standard PCA Analysis')
plt.show()
""" It is visible plot the PCA plot that a linear classifier would not be able to perform well on the data
    transformed by the standard PCA"""

# Now we try Kernel PCA for transformation

X_kpca = rbf_kernel_pca(X,gamma=15,n_components=2)
fig , ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0],X_kpca[y == 0, 1], color='red', marker = '^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0],X_kpca[y == 1, 1], color='blue', marker = 'o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
plt.title('Kernel PCA Aalysis')
plt.show()

# Separating Concentric Circles

from sklearn.datasets import make_circles
X,y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0,0], X[y==0,1], color='red',marker='^',alpha=0.5)
plt.scatter(X[y==1,0], X[y==1,1], color='blue',marker='o',alpha=0.5)
plt.show()

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig , ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_spca[y==0, 0],X_spca[y == 0, 1], color='red', marker = '^', alpha=0.5)
ax[0].scatter(X_spca[y==1, 0],X_spca[y == 1, 1], color='blue', marker = 'o', alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
plt.title('Standard PCA Analysis Concentric Circle Data')
plt.show()

X_kpca = rbf_kernel_pca(X,gamma=15, n_components=2)
fig , ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0],X_kpca[y == 0, 1], color='red', marker = '^', alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0],X_kpca[y == 1, 1], color='blue', marker = 'o', alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC 1')
plt.title('Kernel PCA Aalysis on Concentric Circle Data')
plt.show()
