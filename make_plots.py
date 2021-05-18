from plotter import plot_errors
import os
import sys
from glob import glob
import pickle

files = glob("*.pkl")

for file_ in files:
    print(file_)

    with open(file_, "rb") as f:
        # lists = pickle.load(f)
        # print(len(lists))
        true_nystrom, min_eig_nystrom, our_nystrom, CUR = pickle.load(f)
    f.close()

    nom = file_.split("_")[0]

    if nom == "20ng2":
        nom = "news"

    if nom == "rte" or nom == "mrpc":
        id_count = len(true_nystrom)*10 + 10
    else:
        id_count = 1500
    min_y = 0.0
    max_y = 1.0

    if nom == "news" or nom == "mrpc":
        max_y = 2.0

    plot_errors([true_nystrom, min_eig_nystrom, our_nystrom, CUR], \
                 id_count, \
                 ["true nystrom", "min eig nystrom", "nystrom 1.5-2", "CUR"], \
                 step=10, \
                 colormaps=1, name=nom, \
                 save_path="comparison_all", \
                 y_lims=[min_y, max_y])