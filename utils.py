import os
import glob
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import shutil


def makedirs_ifno(paths):
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(f'{path}')
        os.makedirs(path)


def imshowt(inp, title=None):
    """Imshow for Tensor."""
    inp = torchvision.transforms.functional.to_pil_image(inp)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def imshownp(inp, title=None, savename=None, bgr=False):
    """Imshow for ndarray."""
    if bgr:
        plt.imshow(inp[:, :, ::-1])
    else:
        plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight', dpi=500)
    plt.pause(0.001)  # pause a bit so that plots are updated
