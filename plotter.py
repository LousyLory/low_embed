import matplotlib.pyplot as plt
import numpy as np
import os

def plot_errors(lists, id_count, labels, step=10, colormaps=1, name="MRPC", \
    save_path="comparison_among_eigenvalues_and_z", y_lims=[]):


    x_axis = list(range(10, id_count, step))

    plt.gcf().clear()
    scale_ = 0.55
    new_size = (scale_ * 10, scale_ * 8.5)
    plt.gcf().set_size_inches(new_size)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    plt.rc('axes', titlesize=13)
    plt.rc('axes', labelsize=13)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('legend', fontsize=11)

    STYLE_MAP = {"plot":{"marker":".", "markersize":7, "linewidth":1}}

    for i in range(len(lists)):
        error_pairs = lists[i]
        arr1 = np.array(error_pairs)
        ax1.plot(np.array(x_axis),arr1,\
            label=labels[i], **STYLE_MAP["plot"])

    if colormaps == 1:
        colormap = plt.cm.cool
        colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]

    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])

    title_name = name

    directory = "figures/"+save_path+"/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    filename=name.lower()
    path = os.path.join(directory, filename+".pdf")


    plt.locator_params(axis='x', nbins=6)
    if len(y_lims) > 0:
        plt.ylim(bottom=y_lims[0], top=y_lims[1])
    plt.xlabel("Number of landmark samples")
    plt.ylabel("Average approximation error")
    plt.title(title_name, fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax1.legend(loc='upper right')
    plt.savefig(path)
    plt.gcf().clear()