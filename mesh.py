import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmpose.datasets.pipelines import Compose
from mmpose.models import build_posenet
from mmpose.utils.hooks import OutputHook

from bbox import box2cs, xyxy2xywh, xywh2xyxy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'




class LoadImage:
    """A simple pipeline to load image."""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the img_or_path.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            img = mmcv.imread(results['img_or_path'], self.color_type,
                              self.channel_order)
        elif isinstance(results['img_or_path'], np.ndarray):
            results['image_file'] = ''
            if self.color_type == 'color' and self.channel_order == 'rgb':
                img = cv2.cvtColor(results['img_or_path'], cv2.COLOR_BGR2RGB)
        else:
            raise TypeError('"img_or_path" must be a numpy array or a str or '
                            'a pathlib.Path object')

        results['img'] = img
        return results


def _inference_single_pose_model(model,
                                 img_or_path,
                                 bbox,
                                 dataset,
                                 return_heatmap=False):
    """Inference a single bbox.
    num_keypoints: K
    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        bbox (list | np.ndarray): Bounding boxes (with scores),
            shaped (4, ) or (5, ). (left, top, width, height, [score])
        dataset (str): Dataset name.
        outputs (list[str] | tuple[str]): Names of layers whose output is
            to be returned, default: None
    Returns:
        ndarray[Kx3]: Predicted pose x, y, score.
        heatmap[N, K, H, W]: Model output heatmap.
    """

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    assert len(bbox) in [4, 5]
    center, scale = box2cs(cfg, bbox)

    flip_pairs = None
    if dataset in ('TopDownCocoDataset', 'TopDownOCHumanDataset'):
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                      [13, 14], [15, 16]]
    elif dataset == 'TopDownCocoWholeBodyDataset':
        body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                [15, 16]]
        foot = [[17, 20], [18, 21], [19, 22]]

        face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

        hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                [111, 132]]
        flip_pairs = body + foot + face + hand
    elif dataset == 'TopDownAicDataset':
        flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
    elif dataset == 'TopDownMpiiDataset':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    elif dataset == 'TopDownMpiiTrbDataset':
        flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7],
                      [8, 9], [10, 11], [14, 15], [16, 22], [28, 34], [17, 23],
                      [29, 35], [18, 24], [30, 36], [19, 25], [31,
                                                               37], [20, 26],
                      [32, 38], [21, 27], [33, 39]]
    elif dataset in ('OneHand10KDataset', 'FreiHandDataset', 'PanopticDataset',
                     'InterHand2DDataset'):
        flip_pairs = []
    elif dataset in 'Face300WDataset':
        flip_pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11],
                      [6, 10], [7, 9], [17, 26], [18, 25], [19, 24], [20, 23],
                      [21, 22], [31, 35], [32, 34], [36, 45], [37,
                                                               44], [38, 43],
                      [39, 42], [40, 47], [41, 46], [48, 54], [49,
                                                               53], [50, 52],
                      [61, 63], [60, 64], [67, 65], [58, 56], [59, 55]]

    elif dataset in 'FaceAFLWDataset':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                      [12, 14], [15, 17]]

    elif dataset in 'FaceCOFWDataset':
        flip_pairs = [[0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11],
                      [12, 14], [16, 17], [13, 15], [18, 19], [22, 23]]

    elif dataset in 'FaceWFLWDataset':
        flip_pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27],
                      [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
                      [12, 20], [13, 19], [14, 18], [15, 17], [33,
                                                               46], [34, 45],
                      [35, 44], [36, 43], [37, 42], [38, 50], [39,
                                                               49], [40, 48],
                      [41, 47], [60, 72], [61, 71], [62, 70], [63,
                                                               69], [64, 68],
                      [65, 75], [66, 74], [67, 73], [55, 59], [56,
                                                               58], [76, 82],
                      [77, 81], [78, 80], [87, 83], [86, 84], [88, 92],
                      [89, 91], [95, 93], [96, 97]]
    else:
        raise NotImplementedError()

    # prepare data
    data = {
        'img_or_path':
        img_or_path,
        'center':
        center,
        'scale':
        scale,
        'bbox_score':
        bbox[4] if len(bbox) == 5 else 1,
        'dataset':
        dataset,
        'joints_2d':
        np.zeros((cfg.data_cfg.num_joints, 2), dtype=np.float32),
        'joints_2d_visible':
        np.zeros((cfg.data_cfg.num_joints, 2), dtype=np.float32),
        'pose':
        np.zeros((72), dtype=np.float32),
        'joints_3d':
        np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'joints_3d_visible':
        np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'rotation':
        0,
        'ann_info': {
            'image_size': cfg.data_cfg['image_size'],
            'num_joints': cfg.data_cfg['num_joints'],
            'flip_pairs': flip_pairs
        }
    }
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'].data[0]

    # forward the model
    with torch.no_grad():
        result = model(
            img=data['img'],
            img_metas=data['img_metas'],
            return_loss=False,
            return_heatmap=return_heatmap)
    
    # to understand results look at https://github.com/open-mmlab/mmpose/blob/master/mmpose/models/detectors/mesh.py 
    out = dict(joints_3d=result[0][0],
              pose=result[0][1][0],
              shape=result[0][1][1],
              cam=result[0][2],
              all_boxes=result[1],
              image_path=result[2])
    return out


def inference_mesh_model(model,
                         img_or_path,
                         person_results,
                         bbox_thr=None,
                         format='xywh',
                         dataset='TopDownCocoDataset',
                         return_heatmap=False,
                         outputs=None):
    """Inference a single image with a list of person bounding boxes.
    num_people: P
    num_keypoints: K
    bbox height: H
    bbox width: W
    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str| np.ndarray): Image filename or loaded image.
        person_results (List(dict)): the item in the dict may contain
            'bbox' and/or 'track_id'.
            'bbox' (4, ) or (5, ): The person bounding box, which contains
            4 box coordinates (and score).
            'track_id' (int): The unique id for each human instance.
        bbox_thr: Threshold for bounding boxes. Only bboxes with higher scores
            will be fed into the pose detector. If bbox_thr is None, ignore it.
        format: bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).
        dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.
        return_heatmap (bool) : Flag to return heatmap, default: False
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned, default: None
    Returns:
        list[dict]: The bbox & pose info,
            Each item in the list is a dictionary,
            containing the bbox: (left, top, right, bottom, [score])
            and the pose (ndarray[Kx3]): x, y, score
        list[dict[np.ndarray[N, K, H, W] | torch.tensor[N, K, H, W]]]:
            Output feature maps from layers specified in `outputs`.
            Includes 'heatmap' if `return_heatmap` is True.
    """
    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']

    pose_results = []
    returned_outputs = []

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        for person_result in person_results:
            if format == 'xyxy':
                bbox_xyxy = np.expand_dims(np.array(person_result['bbox']), 0)
                bbox_xywh = xyxy2xywh(bbox_xyxy)
            else:
                bbox_xywh = np.expand_dims(np.array(person_result['bbox']), 0)
                bbox_xyxy = xywh2xyxy(bbox_xywh)

            if bbox_thr is not None:
                assert bbox_xywh.shape[1] == 5
                if bbox_xywh[0, 4] < bbox_thr:
                    continue

            returned_outputs = _inference_single_pose_model(
                model,
                img_or_path,
                bbox_xywh[0],
                dataset,
                return_heatmap=return_heatmap)

            #if return_heatmap:
            #    h.layer_outputs['heatmap'] = heatmap

            returned_outputs['layer_outputs'] = h.layer_outputs


            if format == 'xywh':
                person_result['bbox'] = bbox_xyxy[0]

            pose_results.append(person_result)

    return pose_results, returned_outputs


def get_vertices(model, pred_result):
    smpl_out = model.smpl(betas=torch.Tensor(pred_result['shape']).cuda(),
                          body_pose=torch.Tensor(pred_result['pose'][:, 1:]).cuda(),
                          global_orient=torch.Tensor(pred_result['pose'][:, :1]).cuda(),
                          pose2rot=False)
    return smpl_out.vertices.detach().cpu().numpy()