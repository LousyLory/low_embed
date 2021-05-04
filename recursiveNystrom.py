import numpy as np
import math
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
import random
from scipy.spatial.distance import cdist
from utils import lev_plot
import utils

def is_pos_def(x, tol=1e-8):
    return np.all(np.linalg.eigvals(x) > -tol)

def compute_minEig(SKS, eps=1e-16):
    mult = 1.0
    minEig = min(0, eigh(SKS, eigvals_only=True, subset_by_index=[0,0])) - eps
    return minEig

def rescale_levs(levs):
    levs = levs-min(levs)+1e-4;
    levs = levs / np.sum(levs);
    return levs

def recursiveNystrom(K, s, correction=True, minEig=1e-16, expand_eigs=True, \
    eps=1e-16, accelerated=False, KS_correction=True):
    n = K.shape[0]
    
    if accelerated:
        sLevel = np.ceil(np.sqrt(n*s + s^3)/(4*n))
    else:
        sLevel = s
    
    # start of the algorithm
    oversamp = np.math.log(sLevel)
    k = np.ceil(sLevel/(4*oversamp)).astype(int)
    nLevels = np.int(np.ceil(np.math.log2(n/sLevel)))
    # random permutation for successful uniform samples
    perm = np.random.permutation(n)

    # set up sizes for recursive levels
    lSize = np.zeros(nLevels+1)
    lSize[0] = n
    for i in range(1,nLevels+1):
        lSize[i] = np.ceil(lSize[i-1]/2)
        pass
    lSize = lSize.astype(int)

    # rInd: indices of points selected at previous level of recursion
    # at the base level it is just a uniform sample of ~sLevel points
    samp = list(range(0,lSize[-1]))
    rInd = perm[samp]
    weights = np.ones((len(rInd),1))
    
    # diagonal of the whole matrix is np.diag(K)

    # main recursion, unrolled for efficiency
    for l in range(nLevels-1,-1,-1):
        # indices of current uniform samples
        rIndCurr = perm[0:lSize[l]]
        # sampled kernel
        KS = K[rIndCurr][:, rInd]
        SKS = KS[samp]
        # print("checking for loops:", SKS[0,0], rIndCurr[0], rInd[0])
        SKSn = SKS.shape[0]

        ################### START MIN EIG CORRECTION ###############################################
        if correction == True:
            if expand_eigs == False:
                # compute local minEig
                minEig = compute_minEig(SKS, eps=eps)
            # correct using precomputed minEig
            cols = list(range(SKSn))
            if KS_correction:
                KS[samp, cols] = KS[samp, cols] - minEig
            SKS[cols, cols] = SKS[cols, cols] - minEig
        ########################## END MIN EIG CORRECTION ##########################################
        # print("is SkS PSD:", is_pos_def(SKS))
        
        # optimal lambda for taking O(klogk) samples
        if k >= SKSn:
            # for the rare chance we take less than k samples in a round
            lambda_ = 10e-6
            # don't set to zero for stability issues
        else:
            # lambda_ = ( np.sum(np.diag(SKS)*np.squeeze((weights**2))) - \
            #     np.sum(np.abs(np.real( eigs(SKS*(weights**2), k, ncv=400)[0] ))) ) / k
            # so for indefinite matrices this lambda is increasing a lot!
            # lambda_ = min(lambda_,  compute_minEig(SKS, eps=eps))
            # print(lambda_)
            lambda_ = 1.0
            # for the case when lambda may be set to zero
            if lambda_ < 10e-6:
                lambda_ = 10e-6
            # print(np.diag(SKS))
            # print("sum first:", np.sum(np.diag(SKS)*weights**2), \
                # "sum second:", np.sum(np.abs(np.real( eigs(SKS*(weights**2), k)[0] ))), "lambda:", lambda_)
            pass

        # compute and sample by lambda ridge leverage scores
        if l!=0:
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is sLevel in expectation
            R = np.linalg.pinv(SKS + np.diag(lambda_*np.squeeze(weights**(-2))))
            # max(0,.) helps avoid numerical issues, unnecessary in theory
            levs = np.minimum( np.ones_like(rIndCurr), oversamp*(1/lambda_)*\
                np.maximum(np.zeros_like(rIndCurr), np.diag(K)[rIndCurr]- np.sum((KS@R)*KS, axis=1)) )
            #levs = rescale_levs(levs)
            samp = np.where( np.random.random(lSize[l]) < levs )[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample.
            if len(samp) == 0:
                levs[:] = sLevel/lSize[l]
                samp = random.sample(range(lSize[l]), sLevel)
                pass
            weights = np.sqrt(1 / levs[samp])
        else:
            # on the top level, we sample exactly s landmark points without replacement
            R = np.linalg.pinv(SKS + np.diag(lambda_*np.squeeze(weights**(-2))))
            levs = np.minimum(np.ones_like(rIndCurr), (1/lambda_)*\
                np.maximum(np.zeros_like(rIndCurr), np.diag(K)[rIndCurr]- np.sum((KS@R)*KS, axis=1)) )
            # levs = rescale_levs(levs)
            samp = np.random.choice(n, size=s, replace=False, p=levs/sum(levs))
            pass
        rInd = perm[samp]
        # print("lambda in round: ",lambda_, "and leverage scores are: ", levs, "and length of the samples: ", len(levs))

    # lev_plot(levs, "Twitter")
    C = K[:][:, rInd]
    SKS = C[rInd]
    # correct SKS for min Eig
    SKSn = SKS.shape[0]
    indices = range(SKSn)
    SKS[indices, indices] = SKS[indices, indices] - minEig + eps
    # print("is SKS PSD:", is_pos_def(SKS))
    W = np.linalg.pinv(SKS+(10e-6)*np.eye(s))


    error = np.linalg.norm(K - (C@W)@C.T) / np.linalg.norm(K)

    return C, W, error, minEig


def wrapper_for_recNystrom(similarity_matrix, K, num_imp_samples, runs=1, mode="normal", normalize="rows", \
    expand=False, KS_correction=False):
    eps = 1e-3
    mult = 1.0
    error_list = []
    abs_error_list = []
    n = len(similarity_matrix)
    list_of_available_indices = list(range(len(similarity_matrix)))
    avg_min_eig = 0

    for r in range(runs):
        if mode == "eigI":
            if expand:
                new_num = int(np.sqrt(num_imp_samples*n))
                sample_indices_bar = np.sort(random.sample(
                    list_of_available_indices, new_num))
                min_eig_A = eigh(similarity_matrix[sample_indices_bar][:, sample_indices_bar], \
                    eigvals_only=True, subset_by_index=[0,0])
                min_eig_A = min(0, min_eig_A) - eps
            else:
                pass
        else:
            min_eig_A = 0

        if mode == "eigI":
            pass

        C, W, error, minEig = recursiveNystrom(similarity_matrix, num_imp_samples, \
            correction=True, minEig=mult*min_eig_A, expand_eigs=expand, \
            eps=eps, KS_correction=KS_correction)

        rank_l_K = (C @ W) @ C.T
        if np.iscomplexobj(rank_l_K):
            rank_l_K = np.absolute(rank_l_K)

        if normalize == "rows":
            rank_l_K = utils.row_norm_matrix(rank_l_K)
            similarity_matrix = utils.row_norm_matrix(similarity_matrix)
            pass
        if normalize == "laplacian":
            rank_l_K = utils.laplacian_norm_matrix(similarity_matrix, rank_l_K)
            pass
        if normalize == "original":
            pass

        abs_error = np.linalg.norm(K - (C@W)@C.T) / np.linalg.norm(K)
        error_list.append(error)
        abs_error_list.append(abs_error)
        avg_min_eig += min_eig_A
        if r < runs-1:
            del rank_l_K
        pass

    avg_min_eig = avg_min_eig / len(error_list)
    avg_error = np.sum(np.array(error_list)) / len(error_list)
    avg_abs_error = np.sum(np.array(abs_error_list)) / len(abs_error_list)

    return avg_error, avg_abs_error, avg_min_eig, rank_l_K


# K = np.random.random((1000,600))
# gamma = 40
# K = cdist(K, K)
# K = np.exp(-gamma*K)
# # print(K)
# s = 100

# avg_error, avg_abs_error, avg_min_eig, rank_l_K = \
#     wrapper_for_recNystrom(K, K, s, runs=10, mode="eigI", normalize="original", expand=True)

# print(avg_error, avg_abs_error, avg_min_eig)
