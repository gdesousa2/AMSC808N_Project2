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
# Choose the words that are more distant in one category and another
k2 = 10000

score_F = np.sum(M[ind_Flor,:],0)/len(ind_Flor)
score_I = np.sum(M[ind_Ind,:],0)/len(ind_Ind)
w_F = np.argsort(score_F)[::-1]
w_I = np.argsort(score_I)[::-1]

word_diff = np.zeros(d, dtype='int64')
for i in range(d):
    indF = np.argwhere(w_F == i)
    indI = np.argwhere(w_I == i)

    word_diff[i] = np.abs(indF - indI)
words = np.argsort(word_diff)[::-1][:10000]

# PCA for k-words
M2 = M[:,words]
U2,S2,V2 = svd(M2)
n_comp = 2
PCA_words_k2 = [M2@V2.T[:,i] for i in range(n_comp)]

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