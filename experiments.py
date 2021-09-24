##
%load_ext autoreload
%autoreload 2
##

##
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
##

##
theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
cov = np.array([[3, 0], [0, 0.4]]) @ R
data = np.random.multivariate_normal([0, 0], cov, 100)
plt.scatter(*data.T)
plt.xlim([-4, 4])
plt.ylim([-4, 4])

u, s, _ = np.linalg.svd(data @ data.T)
comp = u@np.diag(s)
plt.quiver(*np.zeros((2, 2)), comp[:,0], comp[:,1], color=['r', 'g'])
##


