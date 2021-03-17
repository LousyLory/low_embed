import numpy as np
import random
import sys
from copy import deepcopy
from utils import is_pos_def
import utils
# from scipy.sparse.linalg import eig
def is_pos_def(x, tol=1e-8):
    return np.all(np.linalg.eigvals(x) > -tol)

def novel_greedy_nystrom(similarity_matrix, num_imp_samples, runs=1):
    """
    simple implementation of:
    1. Farahat, Ahmed, Ali Ghodsi, and Mohamed Kamel. "A novel greedy algorithm for Nystrom approximation." 
    In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 269-277. 2011.

    inputs:
    1. similarity matrix = N x N matrix
    2. num_imp_samples = int containing how many samples to be considered here
    3. number of random runs to ensure error is averaged

    output:
    1. avg_error: avergae of all errors across runs
    """
    error_list = []
    for r in range(runs):

        error = compute_error(reduced_matrix, similarity_matrix)
        error_list.append(error)
        pass
    avg_error = np.sum(np.array(error_list))
    return avg_error


def simple_nystrom(similarity_matrix, K, num_imp_samples, runs=1, mode="normal", normalize="rows", expand=False):
    """
    simple implementation of:
    1. Williams, Christopher KI, and Matthias Seeger. "Using the Nystrom method to speed up kernel machines." 
    In Advances in neural information processing systems, pp. 682-688. 2001.

    for notation check section 3 of:
    2. Farahat, Ahmed, Ali Ghodsi, and Mohamed Kamel. "A novel greedy algorithm for Nystrom approximation."
    In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 269-277. 2011.

    inputs:
    1. similarity similarity_matrix = N x N matrix
    1b. K is the original similarity_matrix (non symmetrized)
    2. num_imp_samples = int containing how many samples to be considered here
    3. number of random runs to ensure error is averaged
    4. mode mode used to compute inverse of the sub matrix at any iteration

    output:
    1. avg_error: avergae of all errors across runs
    """
    eps = 1e-3
    mult = 1.0
    #similarity_matrix = deepcopy(similarity_matrix)
    """
        if is_pos_def(similarity_matrix):
                print("original matrix is PSD")
        else:
                print("original matrix is not PSD")
        """
    error_list = []
    abs_error_list = []
    n = len(similarity_matrix)
    list_of_available_indices = list(range(len(similarity_matrix)))
    avg_min_eig = 0
    """
    optimal rank matrix and error
    """
    #u, s, vh = np.linalg.svd(similarity_matrix, full_matrices=True)


    for r in range(runs):
        sample_indices = np.sort(random.sample(
            list_of_available_indices, num_imp_samples))
        A = similarity_matrix[sample_indices][:, sample_indices]

        if mode == "eigI":
            if expand:
                new_num = int(np.sqrt(num_imp_samples*n))
                sample_indices_bar = np.sort(random.sample(
                    list_of_available_indices, new_num))
                A_bar = similarity_matrix[sample_indices_bar][:, sample_indices_bar]
                # print(np.min(np.linalg.eigvals(A_bar)))
                min_eig_A = min(0, np.min(np.linalg.eigvals(A_bar))) - eps
                # min_eig_A = np.min(np.linalg.eigvals(A_bar)) - eps
                # print("min eig A: ", min_eig_A)
            else:
                min_eig_A = min(0, np.min(np.linalg.eigvals(A))) - eps
            A = A - mult*(min_eig_A * np.eye(len(A)))
            pass
        # print("is SKS PSD:", is_pos_def(A))
        D = similarity_matrix[sample_indices].T

        if mode == "eye":
            inv_A = np.eye(len(A))  # np.linalg.inv(A)
            pass
        if mode == "normal" or "eigI":
            inv_A = np.linalg.inv(A) #- 0.01 * np.eye(len(A)))
            pass
        if mode == "mueye":
            mu = np.mean(np.diagonal(A))
            inv_A = mu * np.eye(len(A))
            pass
        if mode == "poseig":
            Dn, Vn = np.linalg.eig(A)
            sub_mat = deepcopy(Dn[Dn > 0])
            sub_mat = np.reciprocal(sub_mat)
            Dn[Dn > 0] = sub_mat
            Dn[Dn <= 0] = 0
            diag_mat = np.zeros_like(A)
            np.fill_diagonal(diag_mat, Dn)
            inv_A = (Vn @ diag_mat) @ Vn.T
            pass
        
        rank_l_K = (D @ inv_A) @ D.T
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

        error_matrix = similarity_matrix - rank_l_K
        abs_error_matrix = K - rank_l_K

        error = np.linalg.norm(error_matrix) / \
            np.linalg.norm(similarity_matrix)
        error_list.append(error)
        abs_error = np.linalg.norm(abs_error_matrix) / np.linalg.norm(K)
        abs_error_list.append(abs_error)
        avg_min_eig += min_eig_A
        # print("avg min eig: ", avg_min_eig)
        if r < runs-1:
            del rank_l_K
            del error_matrix
            del abs_error_matrix
        pass
    avg_min_eig = avg_min_eig / len(error_list)
    # print("final avg eigen values: ", avg_min_eig)
    avg_error = np.sum(np.array(error_list)) / len(error_list)
    avg_abs_error = np.sum(np.array(abs_error_list)) / len(abs_error_list)

    return avg_error, avg_abs_error, avg_min_eig, rank_l_K


# Nystrom class
class Nystrom_builder():
    def __init__(self, db, num_neighbors, distance_measure="dot_product"):
        self.dataset = db
        self.num_neighbors = num_neighbors
        if distance_measure == "dot_product":
            self.similarity_matrix = np.dot(self.dataset, self.dataset.T)
        self.self_purger()
        if len(self.reduced_matrix) < num_neighbors:
            print("length of the reduced matrices", len(self.reduced_matrix))
            raise ValueError('number of neighbors chosen is less than the size of the reduced matrix')


    def find_matches(self, indices):
        reshaped_preds = self.similarity_matrix
        # needs to be filled in
        all_indices = list(range(len(reshaped_preds)))
        missing_indices = list(set(all_indices) - set(indices))
        # first find the exact copies 
        exact_copies = []
        for i in range(len(missing_indices)):
            to_find = reshaped_preds[missing_indices[i],:]
            match_ids = (reshaped_preds == to_find).all(axis=1).nonzero()
            exact_copies.append(match_ids[0])
        # remove the IDs which occur in missing indices for each copies 
        # (this will leave out the ones left in reshaped_preds)
        for i in range(len(exact_copies)):
            removed_duplicates = list(set(exact_copies[i]) - set(missing_indices))[0]
            exact_copies[i] = removed_duplicates
        return missing_indices, exact_copies
    
    def self_purger(self):
        """
        removes duplicate rows and cols if any and finds a mapping for the remaining
        indices to the original set of indices
        """
        # FIND OUT DUPLICATE ROWS AND ELIMINATE!!
        unique_rows, indices = np.unique(self.similarity_matrix, axis=0, return_index=True)

        # FIND DUPLICATE ROWS AND ID THEM WITH MATCHING ROWS 
        # Duplicate ids == rows in original similarity_matrix which has been removed
        # Matched ids = rows in original similarity_matrix which has a duplicate in duplcate_ids
        duplicate_ids, matched_ids = self.find_matches(indices)

        # save details in self
        self.reduced_matrix = unique_rows
        self.indices_mappings = indices
        return self

    def simple_nystrom(self, runs=1):
        """
        needs modification for muliple runs and then choices, check tomorrow
        simple implementation of:
        1. Williams, Christopher KI, and Matthias Seeger. "Using the Nystrom method to speed up kernel machines." 
        In Advances in neural information processing systems, pp. 682-688. 2001.
        for notation check section 3 of:
        2. Farahat, Ahmed, Ali Ghodsi, and Mohamed Kamel. "A novel greedy algorithm for Nystrom approximation."
        In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, pp. 269-277. 2011.
        inputs:
        1. self = object
        2. number of random runs for training
        output:
        1. global_sample_indices: the sample indices chosen for sampling algo
        2. global_rank_l_K: the approximate matrix
        """
        similarity_matrix = self.reduced_matrix
        K = deepcopy(self.reduced_matrix)
        num_imp_samples = self.num_neighbors
        eps = 1e-1

        #similarity_matrix = deepcopy(similarity_matrix)
        error_list = []
        abs_error_list = []
        global_rank_l_k = []
        global_sample_indices = []
        global_error = 1e+5
        n = len(similarity_matrix)
        list_of_available_indices = list(range(len(similarity_matrix)))
        avg_min_eig = 0

        for r in range(runs):
            sample_indices = np.sort(random.sample(list_of_available_indices, num_imp_samples))
            A = similarity_matrix[sample_indices][:, sample_indices]

            new_num = int(np.sqrt(num_imp_samples*n))
            sample_indices_bar = np.sort(random.sample(list_of_available_indices, new_num))
            A_bar = similarity_matrix[sample_indices_bar][:, sample_indices_bar]
            min_eig_A = min(0, np.min(np.linalg.eigvals(A_bar))) - eps
            # print(min_eig_A)
              
            A = A - (min_eig_A * np.eye(len(A)))
            pass

            D = similarity_matrix[sample_indices].T

            
            inv_A = np.linalg.inv(A) #- 0.01 * np.eye(len(A)))
            pass

            rank_l_K = (D @ inv_A) @ D.T
            if np.iscomplexobj(rank_l_K):
                rank_l_K = np.absolute(rank_l_K)

            error_matrix = similarity_matrix - rank_l_K
            abs_error_matrix = K - rank_l_K

            error = np.linalg.norm(error_matrix) / np.linalg.norm(similarity_matrix)
            
            if error < global_error:
                global_rank_l_k = deepcopy(rank_l_K)
                global_sample_indices = deepcopy(sample_indices)
                global_error = error
            else:
                del rank_l_K
                del error_matrix
                del abs_error_matrix
            pass

        # map the best set of sample indices to the global indices
        global_sample_indices = self.indices_mappings[global_sample_indices]
        return global_sample_indices, global_rank_l_k
