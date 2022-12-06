# DataFusionForRecovery

# data fusion using matrices addition

import spectraltree
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
from sklearn.metrics.cluster import rand_score

import toytree
import toyplot
import toyplot.pdf
import toyplot.svg
import time

# shift to decimal form
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# defined tree parameters

num_taxa = 256   # Number of terminal nodes
jc = spectraltree.Jukes_Cantor()   #set evolution process to the Jukes Cantor model
mutation_rate = jc.p2t(0.9)        #set mutation rate between adjacent nodes to 1-0.9=0.1

#reference_tree = spectraltree.balanced_binary(num_taxa)
#reference_tree = spectraltree.unrooted_pure_kingman_tree(num_taxa)
reference_tree = spectraltree.unrooted_birth_death_tree(num_taxa)


# matrices sum

#stuck several similarity matrices in list, n =  Number of independent samples (sequence length)  
def mat_append(mat_num = 3, n = 50):

    mat_num = list(range(mat_num))

    # create list to store all the laplacians
    mat_list = []
    mat_list_for_concat = []

    # create few laplacians inside mat_list
    for L in mat_num:
        
        observations, taxa_meta = spectraltree.simulate_sequences(n, tree_model=reference_tree, seq_model=jc, mutation_rate=mutation_rate, alphabet="DNA")
        
        mat_list_for_concat.append(observations)
        
        # creat a similarity matrix
        W = spectraltree.JC_similarity_matrix(observations)
        # create a degree matrix
        D = np.diag(np.sum(np.array(W), axis=1))
        # create a laplacian matrix
        L = D - W
        
        mat_list.append(L)

    # sum the matries
    mat_sum = sum(mat_list)
    # find the eigenvalus (e) and eigenvectors (v)
    e, v = np.linalg.eigh(mat_sum)    
    # defind the eigen vector that corespond to the second smallest eigen value as the fiedler vector
    fiedler = v[:,1]
    
    # transform negetive values to 0 and positive to 1 in the fiedler vector
    fiedler_divided = np.where(fiedler >= 0, 1, 0)

    
    
    
    #concat
    columns_concated = np.empty((num_taxa, 0))

    for m in mat_num:       
        observation = mat_list_for_concat[m]
        columns_concated = np.append(columns_concated, observation, axis = 1)

        
    # creat a similarity matrix
    W_c = spectraltree.JC_similarity_matrix(columns_concated)

    # create a degree matrix
    D_c = np.diag(np.sum(np.array(W_c), axis=1))

    # create a laplacian matrix
    L_c = D_c - W_c

    # find the eigenvalus (e) and eigenvectors (v)
    e_c, v_c = np.linalg.eigh(L_c)
        
    # defind the eigen vector that corespond to the second smallest eigen value as the fiedler vector
    fiedler_c = v_c[:,1]
    
    # transform negetive values to 0 and positive to 1 in the fiedler vector
    fiedler_divided_c = np.where(fiedler_c >= 0, 1, 0)
        
        
    return fiedler_divided, fiedler_divided_c


fiedler_divided_real = mat_append(mat_num = 1, n = 50000)[0]

mat_sum_num = list(range(1,40,1))

# comparison

def score_sum():
    
    score_list_sum = []
    score_list_concate = []
    for i in mat_sum_num:       
        fiedler_divided, fiedler_divided_c = mat_append(mat_num = i, n = 50)
        
        score_sum = rand_score(fiedler_divided_real, fiedler_divided)
        score_list_sum.append(score_sum)

        score_concate = rand_score(fiedler_divided_real, fiedler_divided_c)
        score_list_concate.append(score_concate)
        
    return score_list_sum, score_list_concate

score_list_sum, score_list_concate = score_sum()



def scor_iter():
    
    iter_list_sum = []
    iter_list_concat = []
    
    iter_range = range(1,5)
    
    for i in iter_range:
        score_list_sum, score_list_concate = score_sum()
        iter_list_sum.append(score_list_sum)
        iter_list_concat.append(score_list_concate)
    
    return iter_list_sum, iter_list_concat
    
iter_list_sum, iter_list_concat = scor_iter()

iter_list_sum = np.array(iter_list_sum)
iter_list_concat = np.array(iter_list_concat)

fig,axs = plt.subplots(2,2, figsize=(8,8))
for i, ax in enumerate(axs.flatten()):
    ax.scatter(mat_sum_num,iter_list_sum[i,:], marker="o", c='blue', alpha=0.2, label ='Matrices Addition')
    ax.scatter(mat_sum_num,iter_list_concat[i,:], marker=".", c='green', alpha=1, label ='Matrices Concatenating')
    ax.set_xlabel('Matrices Number')
    ax.set_ylabel('Rand Score')
    ax.legend(loc ="lower right")
