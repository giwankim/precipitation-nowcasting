import random
import numpy as np
import matplotlib.pyplot as plt
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize(x, y=None, test=False):
    cmap = plt.cm.get_cmap("RdBu")
    cmap = cmap.reversed()
    if test:
        fig, axes = plt.subplots(1, 4, figsize=(10, 10))
        for i, ax in enumerate(axes):
            img = x[:, :, i]
            ax.imshow(img, cmap=cmap)
    else:
        fig, axes = plt.subplots(1, 5, figsize=(10, 10))
        for i, ax in enumerate(axes[:-1]):
            img = x[:, :, i]
            ax.imshow(img, cmap=cmap)
        axes[-1].imshow(y[:, :, 0], cmap=cmap)
    plt.show()


def radar2precipitation(radar):
    """Convert radar to precipitation."""
    dbz = ((radar - 0.5) / 255.0) * 70 - 10
    z = np.power(10.0, dbz / 10.0)
    r = np.power(z / 200.0, 1.0 / 1.6)
    return r
