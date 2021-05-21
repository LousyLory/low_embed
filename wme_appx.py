from liblinear.liblinearutil import *
import numpy as np
from matplotlib import pyplot as plt
# import torch
import sys
from approximator import nystrom_with_eig_estimate as nystrom
from approximator import nystrom_with_samples as nystrom_test
from approximator import CUR
from approximator import CUR_with_samples as CUR_test
from utils import read_mat_file
from scipy.linalg import  sqrtm
from sklearn.model_selection import StratifiedKFold
# import torch
# import torch.nn as nn
# import torch.optim as optim

from absl import flags
from absl import logging
from absl import app
#import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', "twitter", "twitter or ohsumed or news or recipe")
flags.DEFINE_string('method', "CUR", "method for approximation")
flags.DEFINE_float('lambda_inverse', 1e4, "lambda inverse value")
flags.DEFINE_float('gamma', 0.1, "exp(-dsr/gamma)")
flags.DEFINE_integer('sample_size', 500, "number of samples to be considered")

flags.DEFINE_string('mode', "train", "run mode")

logging.set_verbosity(logging.INFO)

def get_feat(X, indices, k, gamma, approximator, mode="train", \
    samples=None, min_eig=None):
    if approximator == "nystrom":
        if mode == "train":
            KS, SKS, sample_indices, min_eig = nystrom(X, k, \
                                  return_type="decomposed", \
                                  mult=2, eig_mult=1.5, indices=list(indices), \
                                  gamma=gamma)
            feats = KS @ sqrtm(np.linalg.inv(SKS))
            return feats, sample_indices, min_eig
        if mode == "val":
            KS, SKS = nystrom_test(X, indices, samples, min_eig, gamma=gamma)
            feats = KS @ sqrtm(np.linalg.inv(SKS))
            return feats

    if approximator == "CUR":
        if mode == "train":
            C, U, sample_indices = CUR(X,\
                                       k,\
                                       indices=list(indices),\
                                       return_type="decomposed",\
                                       gamma=gamma)
            feats = C @ sqrtm(U)
            return feats, sample_indices, None
        if mode == "val":
            C, U = CUR_test(X, indices, sample_indices=samples, gamma=gamma)
            feats = C @ sqrtm(C)
            return feats


# training
def train_all(X, Y, config):
    # create train and validation splits
    kf = StratifiedKFold(n_splits=config["CV"], random_state=None, shuffle=True)

    valAccu = []
    for train_index, val_index in kf.split(X, Y):
        Y_train, Y_val = Y[train_index], Y[val_index]

        # train features
        train_feats, indices, eig = \
                        get_feat(X, train_index, \
                            config["samples"], \
                            config["gamma"], \
                            config["approximator"])

        # validation features
        val_feats = get_feat(X, val_index, \
                              config["samples"], \
                              config["gamma"], \
                              config["approximator"], \
                              mode="val", \
                              samples=indices, \
                              min_eig=eig)

        # hyperparameters
        s = "-s 2 -e 0.0001 -q -c "+str(config["lambda_inverse"])
        # train model
        model_linear = train(Y_train, train_feats, s)
        # predict on validation
        [_, val_accuracy, _] = predict(Y_val, val_feats, model_linear)
        # validation accuracy
        valAccu.append(val_accuracy[0])

    #wandb.log({"validation_mean":np.mean(valAccu), \
    #    "validation_std":np.std(valAccu)})
    #logging.info("validation_accuracy: %s", np.mean(valAccu))
    return None

# main
def main(argv):
    #wandb.init(project="WME-Nyst and CUR")
    #wandb.config.update(flags.FLAGS)
    #logging.info('Running with args %s', str(argv))

    # get dataset
    dataset = FLAGS.dataset
    if dataset == "ohsumed":
        filename = "oshumed_K_set1.mat"
        version = "v7.3"
    if dataset == "twitter":
        filename = "twitter_K_set1.mat"
        version = "default"
    if dataset == "news":
        filename = "20ng2_new_K_set1.mat"
        version = "v7.3"
    if dataset == "recipe":
        filename = "recipe_trainData.mat"
        version = "v7.3"

    approximator = FLAGS.method
    if approximator not in ["nystrom", "CUR"]:
        print("please choose between nystrom and CUR for approximator")
        return None

    # get EMD matrix
    similarity_matrix, labels = read_mat_file(\
                                    file_="./WordMoversEmbeddings/mat_files/"+filename,\
                                    version=version, return_type="all")

    # set hyper-parameters
    # sample_size_list = range(100,4906, 20)
    # CV = 10
    # gamma_list = [1e-3, 1e-2, 5e-2, 0.10, 0.5, 1.0, 1.5]
    # lambda_inverse = [1e2, 3e2, 5e2, 8e2, 1e3, 3e3, 5e3, \
    #                   8e3, 1e4, 3e4, 5e4, 8e4, 1e5, 3e5, \
    #                   5e5, 8e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11]


    config = {"samples":FLAGS.sample_size,\
              "CV":10, \
              "gamma": FLAGS.gamma,\
              "lambda_inverse":FLAGS.lambda_inverse,\
              "approximator":approximator}

    train_all(similarity_matrix, labels, config)
    return None

if __name__ == "__main__":
    print(FLAGS)
    app.run(main)
