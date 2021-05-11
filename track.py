from mmtrack.apis import inference_vid, inference_mot
from mmcv.video import VideoReader
from tqdm.notebook import trange
from torch import nn
from typing import Dict
import numpy as np


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
