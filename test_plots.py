import matplotlib.pyplot as plt
import numpy as np
from absl import app
from scipy.stats import spearmanr,pearsonr
from utils import read_file, read_mat_file

def main(argv):
    import pickle

    scc_x_axis = [567.671635389328,300.22720193862915,277.6796033382416,321.01755714416504,264.8320572376251,241.44643926620483,216.84782433509827,199.49162244796753,190.36441206932068,151.53035879135132,146.02608823776245]
    scc_y_axis = [0.6224902249884662,0.6186861245248759,0.6180919671743269,0.6179048442334829,0.6172565871182824,0.615331433377679,0.6127546531769262,0.6071807975899141,0.6070910345683863,0.595976757389937,0.580182638773454]
    affinity_x_axis = [122.522]
    affinity_y_axis = [0.5307]
    hac_x_axis = [701.339]
    hac_y_axis = [0.6324]
    scc_y_axis.extend(hac_y_axis)
    scc_x_axis.extend(hac_x_axis)
    scc_y_axis.extend(affinity_y_axis)
    scc_x_axis.extend(affinity_x_axis)
    # metrics = pickle.load(open('/tmp/results.pkl', 'rb'))
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=11)
    STYLE_MAP = {"SCC": {"color": "#4d9221",  "marker": ".", "markersize": 7, 'label': 'SCC', 'linewidth': 1},
                 "HAC": {"color": "#7B3294",  "marker": "^", "markersize": 7, 'label': 'HAC', 'linewidth': 0},
                 "Affinity": {"color": "#f1a340", "marker": "s", "markersize": 7, 'label': 'Affinity', 'linewidth': 0},
                 }
    def plot_me():
        plt.gcf().clear()
        scale_ = 0.45
        new_size = (scale_ * 10, scale_ * 8.5)
        plt.gcf().set_size_inches(new_size)
        scc_pairs = [(x, y) for x, y in zip(scc_x_axis, scc_y_axis)]
        scc_sorted_pairs = sorted(scc_pairs, key=lambda x: x[0])
        arr = np.array(scc_sorted_pairs)
        print(arr)
        plt.plot(arr[:, 0], arr[:, 1], **STYLE_MAP['SCC'])
        hac_pairs = [(x, y) for x, y in zip(hac_x_axis, hac_y_axis)]
        hac_sorted_pairs = sorted(hac_pairs, key=lambda x: x[0])
        arr = np.array(hac_sorted_pairs)
        print(arr)
        plt.plot(arr[:, 0], arr[:, 1], **STYLE_MAP['HAC'])
        aff_pairs = [(x, y) for x, y in zip(affinity_x_axis, affinity_y_axis)]
        aff_sorted_pairs = sorted(aff_pairs, key=lambda x: x[0])
        arr = np.array(aff_sorted_pairs)
        print(arr)
        plt.plot(arr[:, 0], arr[:, 1], **STYLE_MAP['Affinity'])
        # plt.title('Jacc/Node - $\\rho=%01.2f (%01.2f)$, $r=%01.2f (%01.2f)$' % (sp, sppval, p, ppval), {'fontsize': 12})
        plt.xlim(left=0.0, right=1.0)
        plt.ylim(bottom=0.5, top=0.65)
        plt.locator_params(axis='x', nbins=6)
        plt.xlabel("Time (s)")
        plt.ylabel("Dendrogram Purity")
        plt.tight_layout()
        plt.legend(loc='lower right')
        plt.savefig('./zzz_fig_ilsvrc_lrg_interpolation.pdf')
        # plt.savefig('/tmp/zzz_fig_aloi_interpolation.pdf')
        plt.close()
    plot_me()
if __name__ == "__main__":
    app.run(main)
