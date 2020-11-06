import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# project2 implemented functions
from utils import *

# ###########################################################
# AUTHOR: GUILHERME DE SOUSA - guilherme2.desousa@gmail.com #
# Question 4:                                               #
#   CUR Factorization                                       #
#############################################################

# initialize data
M, Mlogic, y = readmatrix()
n, d = M.shape
ind_Flor = np.argwhere(y < 0)[:,0]
ind_Ind = np.argwhere(y > 0)[:,0]


############################
# CHOOSING MOST USED WORDS #
############################
k1 = 10 # choice of k from Problem 3
q1 = 10 # number of words to look up

# Leverage score for k-words
U,S,V = svd(M)
pk1 = np.sum(V[:k1,:]**2, 0)/k1

pi_k1 = pk1.argsort()[::-1][:q1] # largest leverage score
words_k1 = get_word(pi_k1) # words with k-largest leverage score

print(f'{q1}-"largest" words: ', end='')
print(words_k1)

# PCA for k-words
n_comp = 2
PCA_words_k1 = [M@V.T[:,i] for i in range(n_comp)]


# Plotting
plt.figure()
plt.plot(PCA_words_k1[0][ind_Flor], 
    PCA_words_k1[1][ind_Flor], 'b^', label='Florida')
plt.plot(PCA_words_k1[0][ind_Ind], 
    PCA_words_k1[1][ind_Ind], 'rs', label='Indiana')
plt.title(f'PCA {k1} words, 2 components')
plt.ylabel('Component 2')
plt.xlabel('Component 1')
plt.legend()
plt.savefig(f'p4/PCA_2_{k1}_words.png')
# these words are not specific from Indiana or Florida, choose another set #


############################
# CHOOSING IMPORTANT WORDS #
############################
# Remove words that apprear a lot in both documents
# Choose 5000 words that appear a lot in Florida but not in Indiana
# Choose 5000 words that appear a lot in Indiana but not in Florida
cmax = np.max(M,0)/np.sum(Mlogic,0)
cmean = np.sum(M,0)/np.sum(Mlogic,0)
ind_mean_keep = np.argwhere(cmean < cmean[0]*1.5)[:,0]
ind_max_keep = np.argwhere(cmax < cmax[0]*1.5)[:,0]

score_F = np.sum(Mlogic[ind_Flor,:],0)/len(ind_Flor)
score_I = np.sum(Mlogic[ind_Ind,:],0)/len(ind_Ind)
w_F = np.argsort(score_F)[::-1]
w_I = np.argsort(score_I)[::-1]

k2 = 10000
q2 = 5

words_Flor = []
words_Ind = []
for i in range(len(w_F)):
    index = np.argwhere(w_I == w_F[i])[0][0]
    if np.abs(i - index) > 3000:
        words_Flor.append(w_F[i])
        words_Ind.append(index)
#words_Flor = words_Flor
#words_Ind = words_Ind
words = list(set(words_Flor + words_Ind))[:k2]

# Leverage score for 10000-words
M2 = M[:,ind_mean_keep]
U2,S2,V2 = svd(M2)
pk2 = np.sum(V2[:k2,:]**2, 0)/k2
pi_k2 = pk2.argsort()[::-1][:q2] # largest leverage score

# PCA for k-words
n_comp = 2
PCA_words_k2 = [M2@V2.T[:,i] for i in range(n_comp)]
#PCA_words_k2 = [M2[:,pi_k2]@V2.T[pi_k2,i] for i in range(n_comp)]

# Plotting
plt.figure()
plt.plot(PCA_words_k2[0][ind_Flor], 
    PCA_words_k2[1][ind_Flor], 'b^', label='Florida')
plt.plot(PCA_words_k2[0][ind_Ind], 
    PCA_words_k2[1][ind_Ind], 'rs', label='Indiana')
plt.title(f'PCA {k2} words, 2 components')
plt.ylabel('Component 2')
plt.xlabel('Component 1')
plt.legend()
plt.savefig(f'p4/PCA_2_{k2}_words.png')