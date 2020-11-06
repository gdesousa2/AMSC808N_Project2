import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# project2 implemented functions
from utils import *

# ###########################################################
# AUTHOR: GUILHERME DE SOUSA - guilherme2.desousa@gmail.com #
# Question 3:                                               #
#   CUR Factorization                                       #
#############################################################

# initialize data
M, Ml, y = readmatrix()
n, d = M.shape

#####################
# CUR FACTORIZATION #
#####################
k = [2, 5, 7, 10]
a = [1, 2, 4, 6, 8]
N_stat = 10 # statistics

norm_ratio = np.zeros((len(a),len(k)))
normMk = np.zeros(len(k))
for j in range(len(k)):
    U,S,V = svd(M)
    S[k[j]:-1] = 0
    diagS = np.zeros((n,d))
    np.fill_diagonal(diagS,S)
    Mk = U@diagS@V
    normMk[j] = np.sqrt(Frob_Norm(M-Mk))
    for i in range(len(a)):
        for x in range(N_stat):
            CUR, norm = CUR_factorization(M,k[j],a[i])
            norm_ratio[i,j] += np.sqrt(norm)
    norm_ratio[:,j] = norm_ratio[:,j] / (N_stat * normMk[j])

plt.figure()
for j in range(len(a)):
    plt.plot(k, norm_ratio[j,:], '^--', label=f'$a = {a[j]}$')
plt.xlabel('rank $k$', fontsize=12)
plt.ylabel('$\Vert M - CUR \Vert_F / \Vert M - M_k \Vert_F$', fontsize=12)
plt.legend()
plt.savefig('p3/CUR_SVD_trunc.png')

plt.figure()
for j in range(len(a)):
    plt.plot(k, normMk, '^--')
plt.xlabel('rank $k$', fontsize=12)
plt.ylabel('$\Vert M - M_k \Vert_F$', fontsize=12)
plt.savefig('p3/SVD_trunc.png')