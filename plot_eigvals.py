import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io as sio

def read_mat_eig_file(path_):
    path_ = path_+"eigavls_info_*"
    print(path_)
    files = glob.glob(path_)
    print(files)
    for f in range(len(files)):
        print(files[f])
        dataset = files[f].split("_")[-1].split(".")[0]
        print(dataset)
        mat = sio.loadmat(files[f])

        plot(dataset, mat['real_eigenvals'], bool(mat['ipsd'][0]), mat['k'][0][0])



def plot(dataset, eigenvals, ipsd, rank):

    dataset = dataset.title()
    if dataset == "Recipel":
        dataset = "RecipeL"
    if dataset == "Stsb":
        dataset = "STS-B"
    if dataset == "Mrpc":
        dataset = "MRPC"
    if dataset == "Rte":
        dataset = "RTE"
    if dataset == "Oshumed":
        dataset = "Ohsumed"
    if dataset == "kong":
        dataset = "DonkeyKong"
    if dataset == "Sigmoid":
        dataset = "DonkeyKong Sigmoid"
    if dataset == "Tps":
        dataset = "DonkeyKong TPS"
    x_axis = list(range(1,len(eigenvals)+1))

    plt.rc('axes', titlesize=13)
    plt.rc('axes', labelsize=13)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('legend', fontsize=11)

    STYLE_MAP = {"eigenvals": {"color": "#dc143c",  "marker": ".", 'label': 'eigenvalues', 'linewidth': 1},
                 }

    plt.gcf().clear()
    scale_ = 0.55
    new_size = (scale_ * 10, scale_ * 8.5)
    plt.gcf().set_size_inches(new_size)
    eigval_estimate_pairs = [(x, y) for x, y in zip(x_axis, list(np.squeeze(eigenvals)))]
    arr1 = np.array(eigval_estimate_pairs)
    plt.scatter(arr1[:, 0], arr1[:, 1], **STYLE_MAP['eigenvals'])
    plt.figtext(.7, .7, "Is PSD? = "+str(ipsd), fontsize=12)
    plt.figtext(.7, .65, "Rank = "+str(rank), fontsize=12)
    plt.locator_params(axis='x', nbins=6)
    plt.xlabel("Eigenvalue indices")
    plt.ylabel("Eigenvalues")
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.legend(loc='upper right')
    plt.title(dataset, fontsize=13)
    # plt.title("Minimum eigenvalue estimates", fontsize=13)
    # plt.savefig("./test2.pdf")
    plt.savefig("figures/final_"+dataset+"_all_eigenvalue_plot.pdf")
    plt.close()


def read_mat_val_file(path_):
    path_ = path_+"*_validation_plot*"
    print(path_)
    files = glob.glob(path_)
    print(files)

    for f in range(len(files)):
        dataset = files[f].split("_")[0].split("\\")[-1]
        mat = sio.loadmat(files[f])
        plot_validation(dataset, mat['validation_out'])

def plot_validation(dataset, validation):
    dataset = dataset.title()
    if dataset == "Twitter":
        x_axis = list(range(100,1500+20,20))
    if dataset == "Ohsumed":
        x_axis = list(range(100,2500+20,20))
    if dataset == "20News":
        x_axis = list(range(100,4096+20,20))
    if dataset == "Recipe":
        dataset = "Recipe-L"
        x_axis = list(range(100,4096+20,20))

    plt.rc('axes', titlesize=13)
    plt.rc('axes', labelsize=13)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('legend', fontsize=11)

    STYLE_MAP = {"accuracy": {"color": "#FF00FF",  "marker": ".", 'label': 'accuracy', 'linewidth': 1},
                 }

    plt.gcf().clear()
    scale_ = 0.55
    new_size = (scale_ * 10, scale_ * 8.5)
    plt.gcf().set_size_inches(new_size)
    accuracy_pairs = [(x, y) for x, y in zip(x_axis, list(np.squeeze(validation)))]
    arr1 = np.array(accuracy_pairs)
    plt.plot(arr1[:, 0], arr1[:, 1], **STYLE_MAP['accuracy'])
    # plt.figtext(.7, .7, "Is PSD? = "+str(ipsd), fontsize=12)
    # plt.figtext(.7, .65, "Rank = "+str(rank), fontsize=12)
    plt.locator_params(axis='x', nbins=6)
    plt.xlabel("Number of landmarks chosen")
    plt.ylabel("Validation accuracy")
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.legend(loc='upper right')
    plt.title(dataset, fontsize=13)
    # plt.title("Minimum eigenvalue estimates", fontsize=13)
    # plt.savefig("./test2.pdf")
    plt.savefig("figures/final_"+dataset+"_validation_plot.pdf")
    plt.close()

read_mat_eig_file("./WordMoversEmbeddings/")
# read_mat_val_file("./WordMoversEmbeddings/")