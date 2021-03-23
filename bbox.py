import torchvision
import numpy as np
from PIL import Image
import mmcv
from mmcv.video import VideoReader
from utils import makedirs_ifno
from tqdm.auto import trange
from mmtrack.apis import inference_vid, inference_mot
from torch import nn


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


def box2cs(cfg, box):
    """This encodes bbox(x,y,w,h) into (center, scale)
    Args:
        x, y, w, h
    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


def bbox_crop(img, bbox):
    left, top, right, bot = bbox
    
    padding = (bot-top)*0.10
    left = left-padding
    top = top-padding
    right = right+padding
    bot = bot+padding
    
    height = bot-top+padding
    width = right-left+padding
    
    size = max(height, width)
    more_height = size - height
    more_width = size - width
    
    top = int(top - more_height/2)
    height = int(height + more_height)
    left = int(left - more_width/2)
    width = int(width + more_width)
    
    img = torchvision.transforms.functional.resized_crop(Image.fromarray(img), top, left, height, width, (500,500))
    img = mmcv.rgb2bgr(np.array(img))
    return img

def closest(true, guesses):
    return guesses[np.abs(guesses - true).sum(axis=1).argmin()]

def pick_track_result(prev_res, res):
    res = res['track_results'][0]
    res = res[:,1:-1]
    if res.shape[0] > 1:
        return closest(prev_res, res)
    else:
        return res[0]

def _detect(track_model : nn.Module, vid : VideoReader, inf_fun, save_out=None, only_first=None):
    if save_out is not None:
        makedirs_ifno([save_out]) # save_out needs ending /
        
    if only_first is not None:
        frame_count = only_first  
    else:
        frame_count = vid.frame_cnt
        
    bbox_res = np.zeros((1,4))
    for frame_id in trange(frame_count):
        img = vid[frame_id]
        track_results = inf_fun(track_model, img, frame_id)
        left, top, right, bot =  pick_track_result(bbox_res[-1], track_results)
        
        if save_out is not None:
            new_frame = track_model.show_result(img, track_results['track_results'])
            new_frame = bbox_crop(img, (left, top, right, bot))
            mmcv.imwrite(new_frame, f'{save_out}{frame_id:06d}.jpg')

        bbox_res = np.concatenate((bbox_res, np.array([[left, top, right, bot]])))

    bbox_res = bbox_res[1:] #discard initial zeros
    
    return bbox_res
    
def detect_mot(track_model : nn.Module, vid : VideoReader, **kwargs):
    return _detect(track_model, vid, inference_mot, **kwargs)

def detect_vid(track_model : nn.Module, vid : VideoReader, **kwargs):
    return _detect(track_model, vid, inference_vid, **kwargs)
