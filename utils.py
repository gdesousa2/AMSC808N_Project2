import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import svd, solve, pinv, lstsq

# ###########################################################
# AUTHOR: GUILHERME DE SOUSA - guilherme2.desousa@gmail.com #
# UTILS.PY FOR SOLVING PROJECT 2 FOR AMSC808N               #
#############################################################
# k-means algorithm
def k_means(A,k):
    n, d = A.shape
    N_max = 10 # max number of steps
    norm = np.zeros(N_max)

    k_ind = np.random.choice(n,k) # random choice of centroid
    m = [A[i,:] for i in k_ind] # initial centroid

    for stp in range(N_max):
        S = [[] for i in range(k)] # set of points in a given cluster
        for i in range(n):
            sn = np.sum((A[i,:] - m[0])**2)
            ind = 0
            sopt = sn
            for j in range(1,k):
                sn = np.sum((A[i,:] - m[j])**2)
                if sn < sopt:
                    ind = j
                    sopt = sn
            S[ind].append(i)
            norm[stp] += np.sum((A[i,:] - m[ind])**2)
        
        # update centroid
        for j in range(k):
            m[j] = np.zeros(d)
            for i in range(len(S[j])):
                m[j] += A[S[j][i],:]
            m[j] = m[j]/len(S[j])
        
    # create matrix R
    R = np.zeros((k,d))
    for i in range(k):
        R[i,:] = m[i]
    
    # create matrix L
    L = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            L[i,j] = 1 * (i in S[j])
    
    return R, L, norm

# Frobenius trace norm
def Frob_Norm(A,B=[]):
    n, d = A.shape
    s = 0
    if B == []: B = A
    for i in range(n):
        for j in range(d):
            s += A[i,j] * B[i,j]
    return s

# NMF algorithm - Non-negative Matrix Factorization
def NMF(A,k,method='GD',alpha=1e-3):
    n, d = A.shape
    N_max = 20 # max number of steps
    norm = np.zeros(N_max)

    # initial condition for W, H
    W = np.maximum(A[:,:k],np.zeros((n,k)))
    H = 0.01*np.random.rand(k,d)
    np.fill_diagonal(H,1)
    
    err = 1e-14 # error parameter for element-se division
    
    if method == 'GD':
        for i in range(N_max):
            R = A - W @ H
            norm[i] = Frob_Norm(R)
            
            H = H + alpha * W.T @ R
            H = np.maximum(H, np.zeros((k,d)))

            W = W + alpha * R @ H.T
            W = np.maximum(W, np.zeros((n,k)))
    
    if method == 'LS':
        for i in range(N_max):
            R = A - W @ H
            norm[i] = Frob_Norm(R)

            H = (H *( W.T @ A)) / (W.T @ W @ H + err)
            W = (W * (A @ H.T)) / (W @ H @ H.T + err)
    
    if method == 'GDLS':
        for i in range(int(N_max/2)):
            R = A - W @ H
            norm[i] = Frob_Norm(R)
            
            H = H + alpha * W.T @ R
            H = np.maximum(H, np.zeros((k,d)))

            W = W + alpha * R @ H.T
            W = np.maximum(W, np.zeros((n,k)))
        
        for j in range(N_max - int(N_max/2)):
            R = A - W @ H
            norm[j+i-1] = Frob_Norm(R)

            H = (H *( W.T @ A)) / (W.T @ W @ H + err)
            W = (W * (A @ H.T)) / (W @ H @ H.T + err)

    return W@H, norm

# Matrix completion algorithm
def matrix_completion(A,k=-1,method='LR',lbd=1e-3):
    n, d = A.shape
    N_max = 10 # number of max steps
    norm = np.zeros(N_max)

    # Projection matrix
    Omega = 1 * (~np.isnan(A)) # Projection matrix
    AR = A.copy()
    AR[np.isnan(AR)] = 0 # Regularize A so method doesn't crash

    # Low rank factorization method
    if method == 'LR':
        # initial condition
        X = AR[:,:k].copy()
        Y = np.zeros((d,k))
        np.fill_diagonal(Y,1)
        for i in range(N_max):
            norm[i] = Frob_Norm(Omega*(AR - X@Y.T))
            
            for j in range(n):
                YOm = Y
                AOm = AR[j,:]
                #ind_Om = np.argwhere(Omega[j,:] > 0)[0]
                #YOm = Y[ind_Om,:]
                #AOm = AR[j,ind_Om]
                #x = solve(YOm.T@YOm + lbd, YOm.T@AOm)
                x = lstsq(YOm.T@YOm + lbd, YOm.T@AOm)[0]
                X[j] = x
            
            for j in range(d):
                XOm = X
                AOm = AR[:,j]
                #ind_Om = np.argwhere(Omega[:,j] > 0)[0]
                #XOm = X[ind_Om,:]
                #AOm = AR[ind_Om,j]
                #y = solve(XOm.T@XOm + lbd, XOm.T@AOm)
                y = lstsq(XOm.T@XOm + lbd, XOm.T@AOm)[0]
                Y[j] = y
        M = X@Y.T

    # Nuclear norm trick algorithm
    if method == 'NN':
        # initial condition
        if k < 0:
            X = AR[:,:5].copy()
            Y = np.zeros((d,5))
        else:
            X = AR[:,:k].copy()
            Y = np.zeros((d,k))
        np.fill_diagonal(Y,1)
        M = X@Y.T
        for i in range(N_max):
            norm[i] = Frob_Norm(Omega*(AR-M))
            U,S,V = svd(M + Omega*(AR - M))
            if k > 0:
                lbd = S[k-1]
            S = np.maximum(S - lbd, np.zeros(S.shape))
            #S[k:-1] = 0 # forces rank ~ k
            Slb = np.zeros((n,d))
            np.fill_diagonal(Slb, S)

            M = U @ Slb @ V
    
    return M, norm

# CUR algorithm
def CUR_factorization(A,k,a=1):
    c = a*k # number of columns
    r = a*k # number of rows

    def col_sel(A,c):
        U,S,V = svd(A)
        p = np.sum(V.T[:,1:k]**2,1)/k
        ptemp = c*p
        eta = np.random.rand(len(ptemp))
        ind = np.argwhere(eta < ptemp)
        Col = A[:,ind][:,:,0]
        return Col

    C = col_sel(A,c)
    R = col_sel(A.T,r).T
    U = pinv(C) @ A @ pinv(R)
    #U = lstsq(R.T,lstsq(C, A)[0].T)[0].T

    CUR = C @ U @ R
    norm = Frob_Norm(A - CUR)
    return CUR, norm

# readmatrix
def readmatrix():
    A = pd.read_csv('vectors.txt', header=None).values
    B = []
    for i in range(len(A)):
        B = B + A[i][0].strip().split()
    B = np.array(B, dtype=int)
    ind = np.argwhere(B > 100000) # entries of document IDs
    n = len(ind) # the number of documents
    la = len(B)
    I = [i for i in range(la)]
    II = np.setdiff1d(I,ind)
    d = max(B[II]) # the number of words in the dictionary

    M = np.zeros((n,d))
    Ml = np.zeros((n,d))
    y = np.zeros(n)
    for j in range(n):
        i = ind[j][0]
        y[j] = B[i+1]
        if j<n-1:
            iend = ind[j+1][0]-1
        else:
            iend = len(B)
        Ml[j,B[i+2:iend-1:2]-1] = 1
        M[j,B[i+2:iend-1:2]-1] = B[i+3:iend:2]
    
    return M,Ml,y

# get word from features_idx.txt
def get_word(w):
    feat = open('features_idx.txt', 'r')
    feat_words = feat.readlines()[9:]
    words = []
    for i in range(len(w)):
        words.append(feat_words[w[i]].split()[1])
    return words