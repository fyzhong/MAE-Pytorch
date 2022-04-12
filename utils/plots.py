import math
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path



def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def patch2img(predict_patch, masked_indices, patch_nums):
    predict_patch = predict_patch.numpy()
    masked_indices = masked_indices.numpy()
    if len(predict_patch.shape) == 3:
        res = np.zeros((predict_patch.shape[0], patch_nums, predict_patch.shape[-1]))
    elif len(predict_patch.shape) == 2:
        res = np.zeros((1, patch_nums, predict_patch.shape[-1]))
        predict_patch = np.expand_dims(predict_patch, axis=0)
        masked_indices = np.expand_dims(masked_indices, axis=0)
    assert len(predict_patch.shape) == 3

    for i in range(predict_patch.shape[0]):
        res[i, masked_indices[i,:],:] = predict_patch[i, :, :]
    return res

def plot_images(image, target, fname='image.jpg'):
    if isinstance(image, torch.Tensor):
        image = image.cpu().float().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().float().numpy()

    if np.max(image) <= 1:
        image *= 255

    bs, _, h, w = image.shape

    if bs > 1:
        image = image[0]
        target = target[0]

    image = image.transpose(1, 2, 0)
    target = target.transpose(1, 2, 0)

    mosaix = np.full((int(h), int(2 * w), 3), 255, dtype= np.uint8)
    mosaix[:, 0:w, :] = image
    mosaix[:, w:, :] = target

    if fname:
        Image.fromarray(mosaix).save(fname)




def plot_labels(labels, save_dir=''):
    # plot dataset labels
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig(Path(save_dir) / 'labels.png', dpi=200)
    plt.close()

    # seaborn correlogram
    try:
        import seaborn as sns
        import pandas as pd
        x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])
        sns.pairplot(x, corner=True, diag_kind='hist', kind='scatter', markers='o',
                     plot_kws=dict(s=3, edgecolor=None, linewidth=1, alpha=0.02),
                     diag_kws=dict(bins=50))
        plt.savefig(Path(save_dir) / 'labels_correlogram.png', dpi=200)
        plt.close()
    except Exception as e:
        pass




