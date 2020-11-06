import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# project2 implemented functions
from utils import *

# ###########################################################
# AUTHOR: GUILHERME DE SOUSA - guilherme2.desousa@gmail.com #
# Question 1:                                               #
#   1.A) K-Means clustering                                 #
#   1.B) NMF factorization                                  #
#       1.B.i) GD                                           #
#       1.B.ii) Lee-Seung                                   #
#       1.B.iii) GD + Lee-Seung                             #
#############################################################
np.set_printoptions(suppress=True) # supress scientific notation

# reading input data - movie ranking for 39 people
A = pd.read_csv('MovieRankings36.csv', header=None).values

# regularized matrix with nan -> 0
AR = A.copy()
AR[np.isnan(AR)] = 0

# projection matrix
Omega = 1 * (~ np.isnan(A))

# completion matrix
AC, normAC = matrix_completion(A,18,'NN')


######################
# K-MEANS CLUSTERING #
######################
k1 = [2, 3, 4, 5]

norm = [[] for i in range(len(k1))]

for i in range(len(k1)):
    R, L, norm[i] = k_means(AC,k1[i])

# Plotting
plt.figure()
for i in range(len(k1)):
    plt.plot(norm[i], '^--', label=f'{k1[i]} cluster(s)')
plt.xticks([5*x for x in range(3)])
plt.xlabel('Step $i$', fontsize=12)
plt.ylabel('WCS $\sum (C_j - x_{ij})^2$', fontsize=12)
plt.legend()
plt.savefig('p1/k_means_iter.png')

plt.figure()
plt.plot(k1, [norm[i][-1] for i in range(len(k1))], '^--')
plt.xlabel('# Clusters', fontsize=12)
plt.ylabel('WCS $\sum (C_j - x_{ij})^2$', fontsize=12)
plt.xticks(k1)
plt.tight_layout()
plt.savefig('p1/k_means_elbow.png')

# Optimal solution
Ropt, Lopt, normKM_opt = k_means(AC,3)
cluster = [[] for i in range(Lopt.shape[1])]
for i in range(len(cluster)):
    cluster[i] = np.argwhere(Lopt[:,i] == 1)
    print()
    print(f'cluster {i} - {len(cluster[i])}:')
    print(np.array_str(Ropt[i,:],precision=1))
    print(cluster[i])

plt.figure()
for i in range(Ropt.shape[0]):
    plt.plot(np.maximum(Ropt[i,:],0), label=f'Cluster {i+1}')
plt.xlabel('Movies')
plt.ylabel('Rank')
plt.xticks([i in range(1,21,2)])
plt.legend()
plt.tight_layout()
plt.savefig('p1/clusters.png')

#####################################
# NON-NEGATIVE MATRIX FACTORIZATION #
#####################################
k2 = 3
alpha = [1e-4, 5e-4, 1e-3, 1.45e-3]
normGD = [[] for i in range(len(alpha))]
normGDLS = [[] for i in range(len(alpha))]

WH, normLS = NMF(AC, 3, method='LS')
for i in range(len(alpha)):
    WH, normGD[i] = NMF(AC, 3, method='GD', alpha=alpha[i])
    WH, normGDLS[i] = NMF(AC, 3, method='GDLS', alpha=alpha[i])

plt.figure()
for i in range(len(alpha)):
    plt.plot(normGD[i], '^--', color=f'C{i}', label=f'PGD step {alpha[i]}')
plt.title(f'PGD, $k = {k2}$')
plt.xlabel('Step i', fontsize=12)
plt.ylabel('norm $\Vert P_\Omega (A - WH) \Vert_F^2$', fontsize=12)
plt.xticks([i for i in range(0,len(normGD[0]),2)])
plt.legend()
plt.tight_layout()
plt.savefig('p1/NMF_GD.png')

plt.figure()
plt.plot(normGD[2], '^--', color=f'C{0}', label=f'PGD step {alpha[2]}')
plt.plot(normLS, 's--', color=f'C{1}', label=f'Lee-Seung')
for i in range(len(alpha)):
    plt.plot(normGDLS[i], 'o--', color=f'C{i+2}', label=f'PGDLS step {alpha[i]}')
plt.title(f'$k = {k2}$')
plt.xlabel('Step i', fontsize=12)
plt.ylabel('norm $\Vert P_\Omega (A - WH) \Vert_F^2$', fontsize=12)
plt.xticks([i for i in range(0,len(normGD[0]),2)])
plt.legend()
plt.tight_layout()
plt.savefig('p1/NMF_GD_LS_GDLS.png')