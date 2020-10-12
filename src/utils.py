import matplotlib.pyplot as plt
from matplotlib.patches import Circle


CMAP = plt.cm.get_cmap("RdBu")
CMAP = CMAP.reversed()
DAMS = (6071, 6304, 7026, 7629, 7767, 8944, 11107)


def visualize(img, cmap=CMAP, dams=False):
    size = img[:, :, 0].shape[0]
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    for i, ax in enumerate(axes[:-1]):
        ax.imshow(img[:, :, i], cmap=cmap)
    axes[-1].imshow(img[:, :, -1], cmap=cmap)
    if dams:
        for dam in DAMS:
            x = dam // size
            y = dam % size

    plt.show()


# PRECIPITATION FROM RADAR
def pixel2dbz(img):
    dbz = (img - 0.5) * 70 / 255 - 10
    return dbz


def dbz2r(dbz):
    z = 10 ** (dbz / 10)
    r = (z / 200) ** (1.0 / 1.6)
    return r


def pixel2r(img):
    return dbz2r(pixel2dbz(img))