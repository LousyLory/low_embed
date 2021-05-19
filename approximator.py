import numpy as np
from copy import deepcopy
import random
# import torch

def nystrom(similarity_matrix, k, min_eig=0.0, min_eig_mode=False, return_type="error", correct_outer=False):
    """
    compute nystrom approximation
    versions:
    1. True nystrom with min_eig_mode=False
    2. Eigen corrected nystrom with min_eig_mode=True
    2a. KS can be eigencorrected with correct_outer=True
    2b. KS not eigencorrected with correct_outer=False
    """
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample(\
                     list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    if min_eig_mode == True:
        A = A - min_eig*np.eye(len(A))
        if correct_outer == False:
            similarity_matrix_x = deepcopy(similarity_matrix)
        else:
            similarity_matrix_x = deepcopy(similarity_matrix)\
                                  - min_eig*np.eye(len(similarity_matrix))
    else:
        similarity_matrix_x = deepcopy(similarity_matrix)
    KS = similarity_matrix_x[:, sample_indices]
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix)


def ratio_nystrom(similarity_matrix, k, min_eig=0.0, min_eig_mode=False, return_type="error"):
    """
    compute nystrom approximation
    versions:
    1. True nystrom with min_eig_mode=False
    2. Eigen corrected nystrom with min_eig_mode=True
    2a. KS can be eigencorrected with correct_outer=True
    2b. KS not eigencorrected with correct_outer=False
    """
    eps=1e-16
    list_of_available_indices = range(len(similarity_matrix))
    sample_indices = np.sort(random.sample(\
                     list_of_available_indices, k))
    A = similarity_matrix[sample_indices][:, sample_indices]
    if min_eig_mode == True:
        A = A - min_eig*np.eye(len(A))
        similarity_matrix_x = deepcopy(similarity_matrix)
    elif min_eig_mode == False:
        similarity_matrix_x = deepcopy(similarity_matrix)
    else:
        local_min_eig = min(0, np.min(np.linalg.eigvals(A))) - eps
        ratio = min_eig / local_min_eig
        A = (1.0/ratio)*A - np.eye(len(A))
        similarity_matrix_x = deepcopy(similarity_matrix)

    KS = similarity_matrix_x[:, sample_indices]
    if return_type == "error":
        if min_eig_mode == True or min_eig_mode == False:
            return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix)
        else:
            return np.linalg.norm(\
                similarity_matrix - \
                (1/ratio)*KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix)


def nystrom_with_eig_estimate(similarity_matrix, k, return_type="error", \
    scaling=False, mult=0, eig_mult=1, indices=None):
    """
    compute eigen corrected nystrom approximations
    versions:
    1. Eigen corrected without scaling (scaling=False)
    2. Eigen corrected with scaling (scaling=True)
    """
    eps=1e-16
    if indices is not None:
        list_of_available_indices = indices
    else:
        list_of_available_indices = range(len(similarity_matrix))

    sample_indices = np.sort(random.sample(\
                             list_of_available_indices, k))

    A = similarity_matrix[sample_indices][:, sample_indices]
    # estimating min eig in the following block
    if mult == 0:
        large_k = np.int(np.sqrt(k*len(similarity_matrix)))
    else:
        if indices is not None:
            large_k = min(mult*k, len(indices)-1)
        else:
            large_k = min(mult*k, len(similarity_matrix)-1)

    larger_sample_indices = np.sort(random.sample(\
                            list_of_available_indices, large_k))
    Z = similarity_matrix[larger_sample_indices][:, larger_sample_indices]
    min_eig = min(0, np.min(np.linalg.eigvals(Z))) - eps
    min_eig = eig_mult*np.real(min_eig)
    if scaling == True:
        min_eig = min_eig*np.float(len(similarity_matrix))/np.float(large_k)

    A = A - min_eig*np.eye(len(A))
    similarity_matrix_x = deepcopy(similarity_matrix)
    if indices is not None:
        KS = similarity_matrix_x[indices][:, sample_indices]
    else:
        KS = similarity_matrix_x[:, sample_indices]
    
    if return_type == "error":
        return np.linalg.norm(\
                similarity_matrix - \
                KS @ np.linalg.pinv(A) @ KS.T)\
                / np.linalg.norm(similarity_matrix), min_eig
    if return_type == "decomposed":
        return KS, A, sample_indices, min_eig


def nystrom_with_samples(similarity_matrix, indices, samples, min_eig):
    A = similarity_matrix[samples][:, samples]
    A = A - min_eig*np.eye(len(A))
    KS = similarity_matrix[indices][:, samples]
    return KS, A



def CUR(similarity_matrix, k, indices=None, \
    eps=1e-3, delta=1e-14, return_type="error", same=False):
    """
    implementation of Linear time CUR algorithm of Drineas2006 et. al.

    input:
    1. similarity matrix in R^{n,d}
    2. integers c, r, and k

    output:
    1. either C, U, R matrices
    or
    1. CU^+R
    or
    1. error = similarity matrix - CU^+R

    """
    n,d = similarity_matrix.shape
    # setting up c, r, eps, and delta for error bound
    # c = (64*k*((1+8*np.log(1/delta))**2) / (eps**4)) + 1
    # r = (4*k / ((delta*eps)**2)) + 1
    # c = 4*k
    c = k
    r = k
    if c > n:
        c= n
    # r = 4*k
    if r > n:
        r = n
    # print("chosen c, r:", c,r)
    try:
        assert 1 <= c and c <= d
    except AssertionError as error:
        print("1 <= c <= m is not true")
    try:
        assert 1 <= r and r <= n
    except AssertionError as error:
        print("1 <= r <= n is not true")
    try:
        assert 1 <= k and k <= min(c,r)
    except AssertionError as error:
        print("1 <= k <= min(c,r)")

    # using uniform probability instead of row norms
    if indices is None:
        pj = np.ones(d).astype(float) / float(d)
        qi = np.ones(n).astype(float) / float(n)
    else:
        pj = np.ones(indices).astype(float) / float(indices)
        qi = np.ones(indices).astype(float) / float(indices)

    # choose samples
    if indices is not None:
        samples_c = np.sort(np.random.choice(indices, c, replace=False, p = pj))
    else:
        samples_c = np.random.choice(range(d), c, replace=False, p = pj)
    if same:
        samples_r = samples_c
    else:
        samples_r = np.random.choice(range(n), r, replace=False, p = qi)

    # grab rows and columns and scale with respective probability
    samp_pj = pj[samples_c]
    samp_qi = qi[samples_r]
    if indices is not None:
        C = similarity_matrix[indices, samples_c] / np.sqrt(samp_pj*c)
    else:
        C = similarity_matrix[:, samples_c] / np.sqrt(samp_pj*c)
    rank_k_C = C
    # modification works only because we assume similarity matrix is symmetric
    if indices is not None:
        R = similarity_matrix[indices, samples_r] / np.sqrt(samp_qi*r)   
    else:
        R = similarity_matrix[:, samples_r] / np.sqrt(samp_qi*r)
    R = R.T
    psi = C[samples_r, :].T / np.sqrt(samp_qi*r)
    psi = psi.T

    U = np.linalg.pinv(rank_k_C.T @ rank_k_C)
    # i chose not to compute rank k reduction of U
    U = U @ psi.T
    
    if return_type == "decomposed":
        return C, U, samples_c
    if return_type == "approximate":
        return (C @ U) @ R
    if return_type == "error":
        # print(np.linalg.norm((C @ U) @ R))
        relative_error = np.linalg.norm(similarity_matrix - ((C @ U) @ R)) / np.linalg.norm(similarity_matrix)
        return relative_error


def CUR_with_samples(similarity_matrix, indices, samples):
    pj = np.ones(indices).astype(float) / float(indices)
    qi = np.ones(indices).astype(float) / float(indices)

    C = similarity_matrix[indices, samples] / np.sqrt(pj * len(samples))
    psi = similarity_matrix[samples, samples] / np.sqrt(qi*len(samples))
    U = np.linalg.pinv(C.T @ C)
    U = U @ psi
    return C, U
