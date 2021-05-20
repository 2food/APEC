import os
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import shutil
from meva.utils import image_utils, kp_utils
from meva.lib.smpl import SMPL, SMPL_MODEL_DIR
from meva.lib import meva_model
from smplx.lbs import vertices2joints
import torch


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
    assert isinstance(inp, np.ndarray)
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


def speed(joints):
    return np.gradient(joints, axis=0)


def acceleration(joints):
    return np.gradient(speed(joints), axis=0)


def acc_error(pred_joints, target_joints, scales, inv_trans):
    pred_joints = image_utils.normalize_2d_kp(pred_joints, inv=True)
    pred_joints = image_utils.trans_point_seq_2d(pred_joints, inv_trans)
    pred_joints = kp_utils.convert_kps(pred_joints, 'spin', 'coco')
    target_joints = image_utils.normalize_2d_kp(target_joints, inv=True)
    target_joints = image_utils.trans_point_seq_2d(target_joints, inv_trans)
    target_joints = kp_utils.convert_kps(target_joints, 'spin', 'coco')
    pa = acceleration(pred_joints[:, :, :2])
    ta = acceleration(target_joints[:, :, :2])
    pa = np.abs(pa[:, :, 0]) + np.abs(pa[:, :, 1])
    ta = np.abs(ta[:, :, 0]) + np.abs(ta[:, :, 1])
    return (pa - ta) / scales[:, np.newaxis]


def acc_error_extra(pred_joints, target_joints, scales, inv_trans):
    pred_joints = image_utils.normalize_2d_kp(pred_joints, inv=True)
    pred_joints = image_utils.trans_point_seq_2d(pred_joints, inv_trans)
    pa = acceleration(pred_joints[:, :, :2])
    ta = acceleration(target_joints[:, :, :2])
    pa = np.abs(pa[:, :, 0]) + np.abs(pa[:, :, 1])
    ta = np.abs(ta[:, :, 0]) + np.abs(ta[:, :, 1])
    return (pa - ta) / scales[:, np.newaxis]


smpl = SMPL(SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False)
smpl_feet_and_hands = [11, 10, 23, 22]
climb_feet_and_hands = [12, 16, 4, 8]


def get_extra_joints(verts, pred_cam):
    smpl_joints = vertices2joints(smpl.J_regressor, torch.Tensor(verts))
    joints2d = meva_model.projection(smpl_joints, torch.Tensor(pred_cam))
    return joints2d.numpy()[:, smpl_feet_and_hands, :]
