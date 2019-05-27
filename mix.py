import numpy as np
from GPy.kern.src.stationary import Stationary

from GPy.util.linalg import  dpotri
from scipy.linalg import lapack



def dK_dX(kern, X, X2=None, dK_dR=None):
    # in place
    # derivative wrt to the second argument!!!!!!!!!!!!!!!!!!!!!!!!
    # stored sparsely in each column
    B, D = X.shape                  # batch size, input dimension

    if X2 is None:
        X2 = X

    J = X2.shape[0]

    if dK_dR is None:
        dK_dR = np.zeros((B,J,D))

    _Temp = -kern.dK_dr_via_X(X,X2) * kern._inv_dist(X,X2)

    if kern.ARD:
        for d in range(0,D):
            dK_dR[:,:,d] =  _Temp * (X[:,d][:,None]-X2[:,d][None,:]) / (kern.lengthscale[d]**2)
    else:
        for d in range(0,D):
            dK_dR[:,:,d] =  _Temp * (X[:,d][:,None]-X2[:,d][None,:]) / (kern.lengthscale**2)

    return dK_dR


setattr(Stationary, "dK_dX", dK_dX)


def dK_dσ02(kern, X, X2=None):
    # note: dK wrt variance σ0^2, transformation: dK_d0 = dK_dσ02 * 2 * σ02

    if X2 is None:
        X2 = X

    return kern.K(X,X2) / kern.variance


setattr(Stationary, "dK_dσ02", dK_dσ02)


def dK_dσ02_diag(kern, X):

    return np.ones(X.shape[0])


setattr(Stationary, "dK_dσ02_diag", dK_dσ02_diag)




def dK_dl(kern, X, X2=None):
    # note: dK wrt lengthscale ls, transformation: dK_dl = dK_dl * l

    if X2 is None:
        X2 = X

    if kern.ARD:
        B, D = X.shape
        J = X2.shape[0]
        _Temp = -kern.dK_dr_via_X(X,X2) * kern._inv_dist(X,X2)
        dK_dl = np.zeros((B,J,D))
        for d in range(0,D):
            dK_dl[:,:,d] = _Temp * np.square( X[:,d:d+1] - X2[:,d:d+1].T ) / (kern.lengthscale[d]**3)
    else:
        dK_dl = (-kern.dK_dr_via_X(X,X2) * kern._scaled_dist(X,X2) / kern.lengthscale )[:,:,None]

    return dK_dl

setattr(Stationary, "dK_dl", dK_dl)

def dK_dl_diag(kern, X):

    if kern.ARD:
        res = np.zeros(X.shape)
    else:
        res = np.zeros(X.shape[0])

    return res

setattr(Stationary, "dK_dl_diag", dK_dl_diag)


# some helper functions

# inversion and log determinant via cholesky
def inv_logDet(M):

    A = np.ascontiguousarray(M)
    L_M, info = lapack.dpotrf(A, lower=1)

    iM, _ = dpotri(L_M)
    logDetM = 2*sum(np.log(np.diag(L_M)))


    return iM, logDetM

# inversion  via cholesky
def inv_c(M):
    A = np.ascontiguousarray(M)
    L_M, info = lapack.dpotrf(A, lower=1)

    #L_M = np.linalg.cholesky(M)
    iM, _ = dpotri(L_M)

    return iM

# multiplication with 3 matrices (or vectors) of appriate size from left to rigth
def dot3lr(A,B,C):
    return np.dot(np.dot(A,B),C)

# multiplication with 3 matrices (or vectors) of appriate size from rigth to left
def dot3rl(A,B,C):
    return np.dot(A,np.dot(B,C))

# computes the diag of H.T K H
def diag_HtKH(H,K):
    return np.sum( np.dot(K,H) * H, 0)


import GPy
def genData(Ntrain, Ntest=1000, D=1, lengthscale=0.2, sig2_noise=0.1, sig2_0=1.5):
    k = GPy.kern.RBF(input_dim=D,lengthscale=lengthscale)
    k.variance = sig2_0

    # Xtrain = np.random.rand(Ntrain,D)*2.-1.
    Xtrain1 = np.random.rand(int(Ntrain/2),D)*0.9-1.
    Xtrain2 = np.random.rand(int(Ntrain/2),D)*0.9+0.1
    Xtrain = np.vstack((Xtrain1,Xtrain2))
    # Xtrain = Xtrain[np.random.permutation(Ntrain),:]
    Xtest = np.random.rand(Ntest,D)*2.2-1.1
    # Xtest = np.random.rand(Ntest,D)*2.-1.
    inds = np.argsort(Xtest[:,0])
    Xtest = Xtest[inds,:]
    # Xtest = Xtest[np.random.permutation(Ntest),:]

    X = np.vstack((Xtrain,Xtest))
    mu = np.zeros((Ntrain+Ntest)) # vector of the means
    C = k.K(X,X) # covariance matrix
    Z = np.random.multivariate_normal(mu,C,1).transpose()
    Ytrain = Z[0:Ntrain] + np.random.randn(Ntrain,1)*np.sqrt(sig2_noise)

    Ytest = Z[Ntrain:] + np.random.randn(Ntest,1)*np.sqrt(sig2_noise)

    return X[0:Ntrain,:], X[Ntrain:,:], Ytrain, Z[Ntrain:], Ytest
