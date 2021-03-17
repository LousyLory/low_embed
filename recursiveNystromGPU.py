import numpy as np
import math
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
import random
from scipy.spatial.distance import cdist

import torch

def is_pos_def(x, tol=1e-8):
    return (compute_minEig(x)[0] > tol).detach().cpu().numpy()

def compute_minEig(SKS, eps=1e-16):
    mult = 1.0
    minEig = torch.lobpcg(SKS.expand(1, -1, -1), k=1, largest=False, method="ortho")[0]
    # minEig = min(0, eigh(SKS, eigvals_only=True, subset_by_index=[0,0])) - eps
    return minEig

def get_top_k_eigenvalues(matrix_, num_eigs):
    try:
        return torch.lobpcg(matrix_.expand(1, -1, -1), k=num_eigs, largest=True, method="ortho")[0].type(torch.complex64)
    except:
        return torch.sort(torch.symeig(matrix_, eigenvectors=False)[0], descending=True)[0][0:num_eigs].type(torch.complex64)

def rescale_levs(levs):
    levs = levs-torch.min(levs)+1e-4;
    levs = levs / torch.sum(levs);
    return levs

def recursiveNystrom(K, s, correction=True, minEig=1e-16, expand_eigs=True, eps=1e-16, accelerated=False):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n = K.size()[0]
    
    if accelerated:
        sLevel = np.ceil(np.sqrt(n*s + s^3)/(4*n))
    else:
        sLevel = s
    
    # start of the algorithm
    oversamp = np.math.log(sLevel)
    k = np.ceil(sLevel/(4*oversamp)).astype(int)
    nLevels = np.int(np.ceil(np.math.log2(n/sLevel)))
    # random permutation for successful uniform samples
    perm = torch.randperm(n).to(device)

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
    weights = torch.ones([len(rInd),1], dtype=torch.float64).to(device)
    
    # diagonal of the whole matrix is np.diag(K)

    # main recursion, unrolled for efficiency
    for l in range(nLevels-1,-1,-1):
        # indices of current uniform samples
        rIndCurr = perm[0:lSize[l]]
        # sampled kernel
        KS = K[np.ix_(rIndCurr.detach().cpu().numpy(), rInd.detach().cpu().numpy())]
        KS = KS.to(device)
        SKS = KS[samp]
        SKS = SKS.to(device)
        # print("checking for loops:", SKS[0,0], rIndCurr[0], rInd[0])
        SKSn = SKS.size()[0]

        ################### START MIN EIG CORRECTION ###############################################
        if correction == True:
            if expand_eigs == False:
                # compute local minEig
                minEig = compute_minEig(SKS, eps=eps)
            # correct using precomputed minEig
            cols = list(range(SKSn))
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
            lambda_ = ( torch.sum(torch.diag(SKS) * torch.pow(weights, 2)) - \
                torch.sum(torch.abs(torch.real( get_top_k_eigenvalues(SKS*torch.pow(weights, 2), k) ))) ) / k
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
            R = torch.inverse(SKS + torch.diag(lambda_*torch.pow(weights,-2))).to(dtype=KS.dtype).to(device)
            # max(0,.) helps avoid numerical issues, unnecessary in theory
            levs = torch.minimum( torch.ones_like(rIndCurr), oversamp*(1/lambda_)*\
                    torch.maximum(torch.zeros_like(rIndCurr), torch.diag(K)[rIndCurr]- \
                    torch.sum(torch.matmul(KS,R)*KS, dim=1)) )

            levs = rescale_levs(levs)

            samp = np.where( np.random.random(lSize[l]) < levs.detach().cpu().numpy() )[0]

            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample.

            if len(samp) == 0:
                levs[:] = sLevel/lSize[l]
                samp = random.sample(range(lSize[l]), sLevel)
                pass

            weights = torch.sqrt(1 / levs[samp])

        else:

            # on the top level, we sample exactly s landmark points without replacement
            R = torch.inverse(SKS + torch.diag(lambda_*torch.pow(weights,-2))).to(dtype=KS.dtype).to(device)

            levs = torch.minimum(torch.ones_like(rIndCurr), (1/lambda_)*\
                    torch.maximum(torch.zeros_like(rIndCurr), torch.diag(K)[rIndCurr]- \
                    torch.sum(torch.matmul(KS,R)*KS, dim=1)) )

            levs = rescale_levs(levs)

            samp = np.random.choice(n, size=s, replace=False, \
                p=levs.detach().cpu().numpy()/sum(levs.detach().cpu().numpy()))

            pass

        rInd = perm[samp]


    C = K[np.ix_(list(range(len(K))), rInd.detach().cpu().numpy())]
    SKS = C[rInd]
    # correct SKS for min Eig
    SKSn = SKS.size()[0]
    indices = range(SKSn)
    SKS[indices, indices] = SKS[indices, indices] - minEig + eps
    # print("is SKS PSD:", is_pos_def(SKS))
    W = torch.inverse(SKS+(10e-6)*torch.eye(s).to(device))


    error = torch.norm(K - torch.matmul(torch.matmul(C,W), torch.transpose(C, 0, 1))) / torch.norm(K)

    return C, W, error, minEig


def wrapper_for_recNystrom(similarity_matrix, K, num_imp_samples, runs=1, mode="normal", normalize="rows", expand=False):
    eps = 1e-3
    mult = 1.5
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
                min_eig_A = torch.lobpcg(similarity_matrix[np.ix_(sample_indices_bar, sample_indices_bar)].expand(1, -1, -1), \
                    k=1, largest=False, method="ortho")[0]
                min_eig_A = min(0, min_eig_A) - eps
            else:
                pass
        pass
        
        if mode == "eigI":
            pass

        C, W, error, minEig = recursiveNystrom(similarity_matrix, num_imp_samples, \
            correction=True, minEig=mult*min_eig_A, expand_eigs=True, eps=eps)

        # rank_l_K = (C @ W) @ C.T
        rank_l_K = torch.matmul(torch.matmul(C, W), torch.transpose(C, 0, 1))

        # if np.iscomplexobj(rank_l_K):
        #     rank_l_K = np.absolute(rank_l_K)

        # if normalize == "rows":
        #     rank_l_K = utils.row_norm_matrix(rank_l_K)
        #     similarity_matrix = utils.row_norm_matrix(similarity_matrix)
        #     pass
        # if normalize == "laplacian":
        #     rank_l_K = utils.laplacian_norm_matrix(similarity_matrix, rank_l_K)
        #     pass
        if normalize == "original":
            pass

        abs_error = torch.norm(K - rank_l_K) / torch.norm(K)
        error_list.append(error.detach().cpu().numpy())
        abs_error_list.append(abs_error.detach().cpu().numpy())
        avg_min_eig += min_eig_A
        if r < runs-1:
            del rank_l_K
        pass

    avg_min_eig = avg_min_eig / len(error_list)
    avg_error = np.sum(np.array(error_list)) / len(error_list)
    avg_abs_error = np.sum(np.array(abs_error_list)) / len(abs_error_list)
    
    return avg_error, avg_abs_error, avg_min_eig, rank_l_K


# K = np.random.random((1000,600))
# K = torch.from_numpy(K).to(0)
# gamma = 40
# K = torch.cdist(K, K)
# K = torch.exp(-gamma*K)
# # print(K)
# s = 100
# #"""
# avg_error, avg_abs_error, avg_min_eig, rank_l_K = \
#     wrapper_for_recNystrom(K, K, s, runs=10, mode="eigI", normalize="original", expand=True)

# print(avg_error, avg_abs_error, avg_min_eig)
#"""