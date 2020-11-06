import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# project2 implemented functions
from utils import *

# ###########################################################
# AUTHOR: GUILHERME DE SOUSA - guilherme2.desousa@gmail.com #
# Question 2:                                               #
#   2.A) Regularization parameter                           #
#   2.B) Matrix rank                                        #
#############################################################
np.set_printoptions(suppress=True) # supress scientific notation

# reading input data - movie ranking for 39 people
A = pd.read_csv('MovieRankings36.csv', header=None).values

# regularized matrix with nan -> 0
AR = A.copy()
AR[np.isnan(AR)] = 0

# projection matrix
Omega = 1 * (~ np.isnan(A))

############################
# REGULARIZATION PARAMETER #
############################
k1 = 5
lbd1 = [1e-4, 1e-2, 1e0, 1e1, 5e1]

normLR1 = [[] for i in range(len(lbd1))]
normNN1 = [[] for i in range(len(lbd1))]

for i in range(len(lbd1)):
    M,nLR = matrix_completion(A,k1,'LR',lbd1[i])
    normLR1[i] = nLR

    M,nNN = matrix_completion(A,method='NN',lbd=lbd1[i])
    normNN1[i] = nNN

# Plotting
plt.figure()
for i in range(len(lbd1)):
    plt.plot(normLR1[i], '^--', color=f'C{i}', label='$\lambda_{LR}$ = '+f'{lbd1[i]}')
plt.xticks([5*x for x in range(3)])
plt.xlabel('Step $i$', fontsize=12)
plt.ylabel('norm $\Vert P_\Omega (A - M) \Vert_F^2$', fontsize=12)
plt.title(f'Low-Rank, $k = {k1}$')
plt.legend()
plt.savefig(f'p2/low_rank_vs_lambda_k{k1}.png')

plt.figure()
for i in range(len(lbd1)):
    plt.plot(normNN1[i], '^--', color=f'C{i}', label='$\lambda_{NN}$ = '+f'{lbd1[i]}')
plt.xlabel('Step $i$', fontsize=12)
plt.ylabel('norm $\Vert P_\Omega (A - M) \Vert_F^2$', fontsize=12)
plt.xticks([5*x for x in range(3)])
plt.title(f'Nuclear norm')
plt.legend()
plt.savefig(f'p2/NN_vs_lambda.png')

plt.figure()
Om = np.sum(Omega)
plt.plot(lbd1, [np.sqrt(normLR1[i][-1]/Om) for i in range(len(lbd1))], '^--', label='Low-rank')
plt.plot(lbd1, [np.sqrt(normNN1[i][-1]/Om) for i in range(len(lbd1))], 's--', label='Nuclear norm')
plt.xscale('log')
plt.xlabel('$\lambda$', fontsize=12)
plt.ylabel('norm $\Vert P_\Omega (A - M) \Vert_F^2$', fontsize=12)
plt.legend()
plt.savefig(f'p2/LR_NN_vs_lambda.png')



###############
# MATRIX RANK #
###############
k2 = [3, 5, 7, 10, 15, 18]
lbd2 = 1e0

normLR2 = [[] for i in range(len(k2))]
normNN2 = [[] for i in range(len(k2))]

for i in range(len(k2)):
    M,nLR = matrix_completion(A,k2[i],'LR',lbd2)
    normLR2[i] = nLR

    M,nNN = matrix_completion(A,k2[i],'NN',lbd2)
    normNN2[i] = nNN

# Plotting
plt.figure()
for i in range(len(k2)):
    plt.plot(normLR2[i], '^--', color=f'C{i}', label='rank $k$ = '+f'{k2[i]}')
plt.xticks([5*x for x in range(3)])
plt.xlabel('Step $i$', fontsize=12)
plt.ylabel('norm $\Vert P_\Omega (A - M) \Vert_F^2$', fontsize=12)
plt.title(f'Low-rank, $\lambda = {lbd2}$')
plt.legend()
plt.savefig(f'p2/low_rank_vs_k.png')

plt.figure()
for i in range(len(k2)):
    plt.plot(normNN2[i], '^--', color=f'C{i}', label='rank $k$ = '+f'{k2[i]}')
plt.xlabel('Step $i$', fontsize=12)
plt.ylabel('norm $\Vert P_\Omega (A - M) \Vert_F^2$', fontsize=12)
plt.xticks([5*x for x in range(3)])
plt.title(f'Nuclear norm')
plt.legend()
plt.savefig(f'p2/NN_vs_k.png')

plt.figure()
Om = np.sum(Omega)
plt.plot(k2, [np.sqrt(normLR2[i][-1]/Om) for i in range(len(k2))], '^--', label='Low-rank')
plt.plot(k2, [np.sqrt(normNN2[i][-1]/Om) for i in range(len(k2))], 's--', label='Nuclear norm')
plt.xticks(k2)
plt.xlabel('rank $k$', fontsize=12)
plt.ylabel('norm $\Vert P_\Omega (A - M) \Vert_F^2$', fontsize=12)
plt.legend()
plt.savefig(f'p2/LR_NN_vs_k.png')



##############
# PREDICTION #
##############
# Optimal solution
M_opt, norm_opt = matrix_completion(A,18,'NN')
# Prediction 1
Gline = 36
print()
print('Prediction 1')
G_movie_rating = A[Gline,:]
G_ind_unseen = np.argwhere(np.isnan(G_movie_rating))
print(np.array_str(G_movie_rating))
print(np.array_str(M_opt[Gline,:],precision=1))
print()

plt.figure()
plt.plot(G_movie_rating, '^', label='Known entries')
plt.plot(G_ind_unseen, M_opt[Gline,G_ind_unseen], 's', label='Predicted')
plt.xlabel('Movies')
plt.xticks([i for i in range(0,len(G_movie_rating),2)])
plt.ylabel('Ranking')
plt.title(f'User {Gline+1}, NN method rank 18')
plt.legend()
plt.savefig('p2/prediction_1.png')

# Prediction 2
Jline = 16
print()
print('Prediction 2')
J_movie_rating = A[Jline,:]
J_ind_unseen = np.argwhere(np.isnan(J_movie_rating))
print(np.array_str(J_movie_rating))
print(np.array_str(M_opt[Jline,:],precision=1))

plt.figure()
plt.plot(J_movie_rating, '^', label='Known entries')
plt.plot(J_ind_unseen, M_opt[Jline,J_ind_unseen], 's', label='Predicted')
plt.xlabel('Movies')
plt.xticks([i for i in range(0,len(J_movie_rating),2)])
plt.ylabel('Ranking')
plt.title(f'User {Jline+1}, NN method rank 18')
plt.legend()
plt.savefig('p2/prediction_2.png')

# Prediction 3
Xline = 5
print()
print('Prediction 3')
X_movie_rating = A[Xline,:]
X_ind_unseen = np.argwhere(np.isnan(X_movie_rating))
print(np.array_str(X_movie_rating))
print(np.array_str(M_opt[Xline,:],precision=1))

plt.figure()
plt.plot(X_movie_rating, '^', label='Known entries')
plt.plot(X_ind_unseen, M_opt[Xline,X_ind_unseen], 's', label='Predicted')
plt.xlabel('Movies')
plt.xticks([i for i in range(0,len(X_movie_rating),2)])
plt.ylabel('Ranking')
plt.title(f'User {Xline+1}, NN method rank 18')
plt.legend()
plt.savefig('p2/prediction_3.png')