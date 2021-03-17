import numpy as np
import h5py
import os
import requests
import tempfile
import time
from copy import deepcopy
import random

import scann
from Nystrom import Nystrom_builder

# GETTING DATASET
with tempfile.TemporaryDirectory() as tmp:
    response = requests.get("http://ann-benchmarks.com/glove-100-angular.hdf5")
    loc = os.path.join(tmp, "glove.hdf5")
    with open(loc, 'wb') as f:
        f.write(response.content)
    
    glove_h5py = h5py.File(loc, "r")

list(glove_h5py.keys())

dataset = glove_h5py['train']
queries = glove_h5py['test']
print(dataset.shape)
print(queries.shape)

normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]

#### make a small dataset since we have to compute entire pairwise matrix
N = 1000
normalized_dataset_sm = normalized_dataset[0:N, :]


# NYSTROM
searcher = Nystrom_builder(db=normalized_dataset_sm, num_neighbors=100, distance_measure="dot_product")
idx, scann_pred_dist = searcher.simple_nystrom(runs=10)
true_dist = normalized_dataset_sm @ normalized_dataset_sm.T
sqd_error = ((scann_pred_dist - true_dist)**2).sum()
print('nystrom quantization error is %s', sqd_error)

# SCANN
searcher = scann.scann_ops_pybind.builder(db=normalized_dataset_sm, num_neighbors=N, distance_measure="dot_product").score_ah(dimensions_per_block=2,
      anisotropic_quantization_threshold=float("nan"),
      training_sample_size=N,
      min_cluster_size=100,
      hash_type="lut16",
      training_iterations=10).build()
idx, scann_pred_dist = searcher.search_batched_parallel(normalized_dataset_sm)
true_dist = normalized_dataset_sm @ normalized_dataset_sm.T
sqd_error = ((scann_pred_dist - true_dist)**2).sum()
print('scann quantization error is %s', sqd_error)
