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
# import wandb

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', "twitter", "twitter or ohsumed or news or recipe")
flags.DEFINE_string('method', "nystrom", "method for approximation")
flags.DEFINE_float('lambda_inverse', 10, "lambda inverse value")
flags.DEFINE_float('gamma', 1.0, "exp(-dsr/gamma)")
flags.DEFINE_integer('sample_size', 1500, "number of samples to be considered")
flags.DEFINE_string('run_mode', "test", "validation or test mode")

flags.DEFINE_string('mode', "train", "run mode")

# logging.set_verbosity(logging.INFO)

def get_feat(X, indices, k, gamma, approximator, mode="train", \
    samples=None, min_eig=None, SKS_prev=None, U_prev=None, S_prev=None):
    if approximator == "nystrom":
        if mode == "train":
            print("log check:", X.shape, k)
            KS, SKS, sample_indices, min_eig = nystrom(X, k, \
                                  return_type="decomposed", \
                                  mult=2, eig_mult=1.5, indices=list(indices), \
                                  gamma=gamma)
            feats = KS @ sqrtm(np.linalg.inv(SKS))
            return feats, sample_indices, min_eig, SKS
        if mode == "val":
            print("log check:", X.shape, k)
            KS, SKS = nystrom_test(X, indices, samples, min_eig, gamma=gamma)
            if SKS_prev is None:
                feats = KS @ sqrtm(np.linalg.inv(SKS))
            else:
                feats = KS @ sqrtm(np.linalg.inv(SKS_prev))
            return feats

    if approximator == "CUR":
        if mode == "train":
            C, U, sample_indices = CUR(X, k,\
                                       indices=list(indices),\
                                       return_type="decomposed",\
                                       gamma=gamma)
            # avoid U being singular
            U = U+eps*np.eye(U.shape)
            u, s, vh = np.linalg.svd(np.linalg.inv(U), full_matrices=True)
            feats = C @ (u @ sqrtm(np.diag(s)))
            
            return feats, sample_indices, u, s
        if mode == "val":
            C, U = CUR_test(X, indices, samples=samples, gamma=gamma)
            if U_prev is None:
                # avoid U being singular
                U = U+eps*np.eye(U.shape)
                u, s, vh = np.linalg.svd(np.linalg.inv(U), full_matrices=True)
            else:
                u = U_prev
                s = S_prev
            feats = C @ (u @ sqrtm(np.diag(s)))
            return feats


# training
def train_all(X, Y, config, X_test=None, Y_test=None):
    # create train and validation splits
    kf = StratifiedKFold(n_splits=config["CV"], random_state=None, shuffle=True)

    valAccu = []
    for train_index, val_index in kf.split(X, Y):
        Y_train, Y_val = Y[train_index], Y[val_index]
        if config["run_mode"] == "test":
            Y_val = Y_test

        # train features
        if config["approximator"] == "nystrom":
            train_feats, indices, eig, SKS = \
                            get_feat(X, train_index, \
                                config["samples"], \
                                config["gamma"], \
                                config["approximator"])

            # validation features
            if config["run_mode"] == "validate":
                val_feats = get_feat(X, val_index, \
                                      config["samples"], \
                                      config["gamma"], \
                                      config["approximator"], \
                                      mode="val", \
                                      samples=indices, \
                                      min_eig=eig, \
                                      SKS_prev=SKS)
            if config["run_mode"] == "test":
                val_feats = get_feat(X_test, list(range(len(X_test))), \
                                      config["samples"], \
                                      config["gamma"], \
                                      config["approximator"], \
                                      mode="val", \
                                      samples=indices, \
                                      min_eig=eig, \
                                      SKS_prev=SKS)
        else:
            train_feats, indices, U_train, S_train = \
                            get_feat(X, train_index, \
                                config["samples"], \
                                config["gamma"], \
                                config["approximator"])

            # validation features 
            if config["run_mode"] == "validate":
                val_feats = get_feat(X, val_index, \
                                      config["samples"], \
                                      config["gamma"], \
                                      config["approximator"], \
                                      mode="val", \
                                      samples=indices, \
                                      U_prev=U_train,\
                                      S_prev=S_train)
            if config["run_mode"] == "test":
                val_feats = get_feat(X_test, list(range(len(X_test))), \
                                      config["samples"], \
                                      config["gamma"], \
                                      config["approximator"], \
                                      mode="val", \
                                      samples=indices, \
                                      U_prev=U_train,\
                                      S_prev=S_train)

        # hyperparameters
        s = "-s 2 -e 0.0001 -q -c "+str(config["lambda_inverse"])
        # train model
        model_linear = train(Y_train, train_feats, s)
        # predict on validation
        [_, val_accuracy, _] = predict(Y_val, val_feats, model_linear)
        # validation accuracy
        valAccu.append(val_accuracy[0])
    print(np.mean(valAccu), np.std(valAccu))
    # wandb.log({"validation_mean":np.mean(valAccu), \
    #    "validation_std":np.std(valAccu)})
    # logging.info("validation_accuracy: %s", np.mean(valAccu))
    return None

# main
def main(argv):
    # wandb.init(project="WME-Nyst and CUR")
    # wandb.config.update(flags.FLAGS)
    # logging.info('Running with args %s', str(argv))

    # get dataset
    dataset = FLAGS.dataset
    if dataset == "ohsumed":
        filename = "oshumed_K_set1.mat"
        if FLAGS.run_mode == "test":
            test_filename = "oshumed_K_set1.mat"
        version = "v7.3"
    if dataset == "twitter":
        filename = "twitter_K_set1.mat"
        if FLAGS.run_mode == "test":
            test_filename = "twitter_K_set1.mat"
        version = "default"
    if dataset == "news":
        filename = "20ng2_new_K_set1.mat"
        if FLAGS.run_mode == "test":
            test_filename = "20ng2_new_K_set1.mat"
        version = "v7.3"
    if dataset == "recipe":
        filename = "recipe_trainData.mat"
        if FLAGS.run_mode == "test":
            test_filename = "recipe_K_set1"
        version = "v7.3"

    approximator = FLAGS.method
    if approximator not in ["nystrom", "CUR"]:
        print("please choose between nystrom and CUR for approximator")
        return None

    # get EMD matrix
    # similarity_matrix, labels = read_mat_file(\
    #                                 file_="/mnt/nfs/work1/elm/ray/"+filename,\
    #                                 version=version, return_type="all")
    similarity_matrix, labels = read_mat_file(\
                                    file_="./WordMoversEmbeddings/mat_files/"+filename,\
                                    version=version, return_type="all")

    # set hyperparameters
    config = {"samples":FLAGS.sample_size,\
              "CV":10, \
              "gamma": FLAGS.gamma,\
              "lambda_inverse":FLAGS.lambda_inverse,\
              "approximator":approximator,
              "run_mode":FLAGS.run_mode}

    if config["run_mode"] == "test":
        test_sim_mat, test_labels = read_mat_file(\
                                                file_="./WordMoversEmbeddings/mat_files/"+test_filename,\
                                                version=version, return_type="all", mode="test")

    if config["run_mode"] == "validate":
        train_all(similarity_matrix, labels, config)
    else:
        train_all(similarity_matrix, labels, config, \
            X_test=test_sim_mat, Y_test=test_labels)
    return None

if __name__ == "__main__":
    # print(FLAGS)
    app.run(main)
