import numpy as np

corr = np.array([[1,0.7,0.4],[0.7,1,0.9],[0.4,0.9,1]])
diag_sigma = np.array([[0.12,0,0],[0,0.18,0],[0,0,0.26]])
big_sigma = np.dot(np.dot(diag_sigma, corr), diag_sigma)
print 'Big Sigma: ',big_sigma
print ""
eigenvalues, eigenvectors = np.linalg.eig(big_sigma)
print 'Eigenvalues', eigenvalues