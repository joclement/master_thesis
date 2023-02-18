import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import tslearn.metrics


def main(filepath: str = "warping_matrix.pdf"):
    a = np.array([0, 0, 0, 0, 1, 2, 3, 2, 0, 1, 1, 1, 2, 3, 4, 1]).reshape((-1, 1))
    b = np.array([0, 1, 2, 3, 1, 0, 0, 1, 1, 2, 3, 4, 1, 1, 1, 1]).reshape((-1, 1))
    sz = a.shape[0]

    path, sim = tslearn.metrics.dtw_path(a, b)

    plt.figure(1, figsize=(8, 8))

    # definitions for the axes
    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02

    rect_s_y = [left, bottom, w_ts, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]
    rect_gram = [left_h, bottom, width, height]
    rect_colorbar = [left_h + width + 0.01, bottom, 0.025, height]

    ax_gram = plt.axes(rect_gram)
    ax_colorbar = plt.axes(rect_colorbar)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    mat = cdist(a, b)

    im = ax_gram.imshow(mat, origin="lower")
    ax_gram.autoscale(False)
    ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-", linewidth=3.0)
    plt.colorbar(im, cax=ax_colorbar)

    ax_s_x.plot(np.arange(sz), b, linewidth=3.0)
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, sz - 1))

    ax_s_y.plot(-a, np.arange(sz), linewidth=3.0)
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, sz - 1))

    plt.tight_layout()
    plt.savefig(fname=filepath)
