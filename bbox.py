import torchvision
import numpy as np
from PIL import Image
import mmcv
from mmcv.video import VideoReader
from utils import makedirs_ifno
from tqdm.notebook import trange
from mmtrack.apis import inference_vid, inference_mot
from torch import nn
from skimage import img_as_float
from typing import Dict


def xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.
    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1
    return bbox_xywh


def xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.
    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0] - 1
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1] - 1
    return bbox_xyxy


def cxcywh2xyxy(bbox_cxcywh):
    """Transform the bbox format from xywh to x1y1x2y2.
    Agrs:
        bbox_cxcywh (ndarray): Bounding boxes,
            shaped (n, 4) or (n, 5). (center x, center y, width height, [score])
    Returns:
        np.ndarray: Bounding boxes, shaped (n, 4) or (n, 5). (left, top, right, bottom, [score]).
    """
    bbox_xyxy = bbox_cxcywh.copy()
    bbox_xyxy[:, 0] = bbox_cxcywh[:, 0] - bbox_cxcywh[:, 2] / 2
    bbox_xyxy[:, 1] = bbox_cxcywh[:, 1] - bbox_cxcywh[:, 3] / 2
    bbox_xyxy[:, 2] = bbox_cxcywh[:, 0] + bbox_cxcywh[:, 2] / 2
    bbox_xyxy[:, 3] = bbox_cxcywh[:, 1] + bbox_cxcywh[:, 3] / 2
    return bbox_xyxy


def cxcywh2cs(bbox_cxcywh, input_size):
    """Transform the bbox format from xywh to (center,scale).
    Agrs:
        bbox_cxcywh (ndarray): Bounding boxes,
            shaped (n, 4) (center x, center y, width, height)
    Returns:
         np.ndarray(n, 4): Center x y, scale x y.
    """
    bs = xyxy2xywh(cxcywh2xyxy(bbox_cxcywh))
    return np.array([box2cs(b, input_size) for b in bs]).reshape(len(bs),4)


def box2cs(box, input_size):
    """This encodes bbox(x,y,w,h) into (center, scale)
    Args:
        x, y, w, h
    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    return center, scale


def bbox_crop(img, bbox):
    left, top, right, bot = bbox

    padding = (bot - top) * 0.10
    left = left - padding
    top = top - padding
    right = right + padding
    bot = bot + padding

    height = bot - top + padding
    width = right - left + padding

    size = max(height, width)
    more_height = size - height
    more_width = size - width

    top = int(top - more_height / 2)
    height = int(height + more_height)
    left = int(left - more_width / 2)
    width = int(width + more_width)

    img = torchvision.transforms.functional.resized_crop(
        Image.fromarray(img), top, left, height, width, (500, 500))
    img = mmcv.rgb2bgr(np.array(img))
    return img


def closest(true, guesses):
    return guesses[np.abs(guesses - true).sum(axis=1).argmin()]


def format_track_result(this_res: Dict):
    if 'track_results' in this_res:
        bbs = this_res['track_results'][0]
        res = {}
        for bb in bbs:
            idx = int(bb[0])
            res[idx] = bb[1:]
        return res
    else:
        return {}


def _detect(track_model: nn.Module, vid: VideoReader, inf_fun, only_first=None):
    if only_first is not None:
        frame_count = only_first
    else:
        frame_count = vid.frame_cnt

    bbox_res = []
    for frame_id in trange(frame_count):
        img = vid[frame_id]
        track_results = inf_fun(track_model, img, frame_id)
        results = format_track_result(track_results)
        bbox_res.append(results)

    return bbox_res


def detect_mot(track_model: nn.Module, vid: VideoReader, **kwargs):
    return _detect(track_model, vid, inference_mot, **kwargs)


def detect_vid(track_model: nn.Module, vid: VideoReader, **kwargs):
    return _detect(track_model, vid, inference_vid, **kwargs)


def seperate_tracks(bboxes):
    highest_idx = max([max(b.keys()) for b in bboxes])
    tracks = np.zeros((len(bboxes), highest_idx + 1, 5))
    for i, b in enumerate(bboxes):
        for k in b.keys():
            tracks[i, k] = b[k]
    return tracks


def get_tracks_with_contigous(tracks, n):
    nonzero = np.all(tracks != 0.0, axis=2)
    return tracks[:, nonzero.sum(axis=0) > n]
