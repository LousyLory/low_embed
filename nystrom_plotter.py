from plotter import plot_errors
import os
import sys
from glob import glob
import pickle

files = glob("*.pkl")

mode = "single plot"

for file_ in files:
    print(file_)

    with open(file_, "rb") as f:
        # lists = pickle.load(f)
        # print(len(lists))
        all_data = pickle.load(f)
        if len(all_data) == 4:
            true_nystrom, min_eig_nystrom, our_nystrom, CUR = all_data
        else:
            true_nystrom, CUR = all_data
    f.close()

    nom = file_.split("_")[0]

    if nom == "20ng2":
        nom = "news"

    if nom == "rte" or nom == "mrpc" or nom == "PSD":
        id_count = len(true_nystrom)*10 + 10
    else:
        id_count = 1500
    min_y = 0.0
    if nom == "news":
        min_y = 0.2
    max_y = 0.5

    if nom == "news" or nom == "mrpc" or nom == "recipe":
        max_y = 0.5

    if nom != "PSD" and mode != "single plot":
        plot_errors([true_nystrom, min_eig_nystrom, our_nystrom, CUR], \
                     id_count, \
                     ["true nystrom", "min eig nystrom", "nystrom 1.5-2", "CUR"], \
                     step=10, \
                     colormaps=1, name=nom, \
                     save_path="comparison_all", \
                     y_lims=[min_y, max_y])
    else:
        if nom == "PSD":
            plot_errors([true_nystrom], id_count,\
                        ["true nystrom"], \
                        step=10,\
                        name=nom,\
                        save_path="true_nystrom")
        if mode == "single plot" and nom == "twitter":
            plot_errors([true_nystrom], id_count,\
                        ["true nystrom"], \
                        step=10,\
                        name=nom,\
                        save_path="true_nystrom")
        if mode == "single plot" and nom == "stsb":
            plot_errors([true_nystrom], id_count,\
                        ["true nystrom"], \
                        step=10,\
                        name=nom,\
                        save_path="true_nystrom")
        if mode == "single plot" and nom == "mrpc":
            plot_errors([true_nystrom], id_count,\
                        ["true nystrom"], \
                        step=10,\
                        name=nom,\
                        save_path="true_nystrom")
        if mode == "true nys vs CUR":
            plot_errors([true_nystrom, CUR], id_count,\
                        ["true nystrom", "CUR"], \
                        step=10,\
                        name=nom,\
                        save_path="true_nystrom")