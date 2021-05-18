import os
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import shutil
from meva.utils import image_utils, kp_utils


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
    ax = plt.axes()
    img = inp
    if bgr:
        img = inp[:, :, ::-1]
    plt.imshow(img)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight', dpi=500)
    return ax


def oks(pred_j2d, target_j2d, scales, inv_trans):
    kpt_oks_sigmas = np.array(
        [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    k2 = kpt_oks_sigmas**2
    s2 = scales**2
    s2 = np.repeat(s2[:, np.newaxis], k2.shape[0], axis=1)

    pred_j2d = np.concatenate(
        (pred_j2d, np.ones(pred_j2d.shape[:2] + (1,))), axis=2)
    pred_j2d = kp_utils.convert_kps(pred_j2d, 'spin', 'coco')
    pred_j2d = image_utils.normalize_2d_kp(pred_j2d, inv=True)
    pred_j2d = image_utils.trans_point_seq_2d(pred_j2d, inv_trans)

    target_j2d = kp_utils.convert_kps(target_j2d, 'spin', 'coco')
    target_j2d = image_utils.normalize_2d_kp(target_j2d, inv=True)
    target_j2d = image_utils.trans_point_seq_2d(target_j2d, inv_trans)

    d = np.linalg.norm(pred_j2d[:, :, :2] - target_j2d[:, :, :2], axis=2)

    return np.exp(-d / (2 * s2 * k2))


def oks_extra(pred_j2d, target_j2d, scales, inv_trans):
    kpt_oks_sigmas = np.array([.89, .89, .62, .62]) / 10
    k2 = kpt_oks_sigmas**2
    s2 = scales**2
    s2 = np.repeat(s2[:, np.newaxis], k2.shape[0], axis=1)

    # pred_j2d = np.concatenate(
    #     (pred_j2d, np.ones(pred_j2d.shape[:2] + (1,))), axis=2)

    pred_j2d = image_utils.normalize_2d_kp(pred_j2d, inv=True)
    pred_j2d = image_utils.trans_point_seq_2d(pred_j2d, inv_trans)

    d = np.linalg.norm(pred_j2d[:, :, :2] - target_j2d[:, :, :2], axis=2)

    return np.exp(-d / (2 * s2 * k2))
