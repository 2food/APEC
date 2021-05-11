from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional
import mmcv
import os
import xml.etree.ElementTree as ET
import numpy as np
from skimage.util import view_as_windows
from meva.utils import image_utils, kp_utils
from tqdm.auto import tqdm


class VideoFrameFolder(ImageFolder):
    """A data loader for sequential images where the samples are arranged in this way: ::

        root/vis_x/0.ext
        root/vid_x/1.ext
        root/vid_x/[...].ext

        root/vid_y/0.ext
        root/vid_y/1.ext
        root/vid_y/[...].ext

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.videos = self.classes
        self.imgs = self.samples
        # sort files, first by video then numerically
        self.imgs.sort(key=lambda x: (
            x[1], int(x[0].split('/')[-1].split('.')[0])))


def read_cvat_anno(file):
    tree = ET.parse(file)
    root = tree.getroot()

    no_frames = int(root.find('meta/task/size').text)
    data = np.empty((no_frames, 19, 2))

    for track in root.findall('track'):
        label = track.attrib['label']
        i = get_climb_cvat_joint_names().index(label)
        for j, point in enumerate(track.findall('points')):
            p_strs = point.attrib['points'].split(',')
            data[j, i] = list(map(float, p_strs))
    return data


def get_climb_cvat_joint_names():
    return [
        'Nose',
        'Right Shoulder',
        'Right Elbow',
        'Right Wrist',
        'Right Hand',
        'Left Shoulder',
        'Left Elbow',
        'Left Wrist',
        'Left Hand',
        'Right Hip',
        'Right Knee',
        'Right Ankle',
        'Right Foot',
        'Left Hip',
        'Left Knee',
        'Left Ankle',
        'Left Foot',
        'Right Ear',
        'Left Ear'
    ]


def get_climb_joint_names():
    return [
        'nose',
        'rshoulder',
        'relbow',
        'rwrist',
        'rhand',
        'lshoulder',
        'lelbow',
        'lwrist',
        'lhand',
        'rhip',
        'rknee',
        'rankle',
        'rfoot',
        'lhip',
        'lknee',
        'lankle',
        'lfoot',
        'rear',
        'lear'
    ]


def get_mmpose_joint_names():
    return [
        "OP Nose",      # 0
        "OP LEye",      # 1
        "OP REye",      # 2
        "OP LEar",      # 3
        "OP REar",      # 4
        "OP LShoulder",  # 5
        "OP RShoulder",  # 6
        "OP LElbow",    # 7
        "OP RElbow",    # 8
        "OP LWrist",    # 9
        "OP RWrist",    # 10
        "OP LHip",      # 11
        "OP RHip",      # 12
        "OP LKnee",     # 13
        "OP RKnee",     # 14
        "OP LAnkle",    # 15
        "OP RAnkle",    # 16
    ]


kp_utils.get_climb_joint_names = get_climb_joint_names
kp_utils.get_mmpose_joint_names = get_mmpose_joint_names

video_names = ['IMG_2139.MOV',
               'IMG_2140.MOV',
               'IMG_2141.MOV',
               'IMG_2142.MOV',
               'IMG_2320.mov',  # side traversal
               'VID_20210123_091729.mp4',
               'VID_20210123_104706.mp4',
               'VID_20210123_110129.mp4',
               'VID_20210123_111337.mp4',
               'VID_20210123_111921.mp4',
               'IMG_2272.mov',
               'IMG_2273.mov',
               'IMG_2274.mov',
               'IMG_2275.mov',
               'IMG_2276.mov',
               'IMG_2277.mov',
               'IMG_2278.mov',
               'IMG_2279.mov',
               'IMG_2280.mov',
               'IMG_2283.mov',
               'IMG_2284.mov',
               'IMG_2285.mov',
               'IMG_2286.mov',
               'IMG_2287.mov',
               'IMG_2288.mov',
               'IMG_2289.mov',
               'IMG_2290.mov',
               'IMG_2292.mov',
               'IMG_2293.mov',
               'IMG_2294.mov',
               'IMG_2295.mov',
               'IMG_2296.mov',
               'IMG_2297.mov',
               'IMG_2298.mov',
               'IMG_2299.mov',
               'IMG_2300.mov',
               'IMG_2301.mov',
               'IMG_2302.mov',
               'IMG_2303.mov',
               'IMG_2304.mov',  # long jump
               'IMG_2305.mov',
               'IMG_2306.mov',  # figure four
               'IMG_2307.mov',
               'IMG_2309.mov',
               'IMG_2310.mov',
               'IMG_2311.mov',
               'IMG_2312.mov',
               'IMG_2313.mov',
               'IMG_2314.mov',  # high step
               'IMG_2315.mov',  # heel hook
               'IMG_2317.mov',
               'IMG_2319.mov',
               'IMG_2321.mov',
               'IMG_2322.mov',
               'VID_20210414_140919.mp4',
               'VID_20210414_140936.mp4',
               'VID_20210414_140957.mp4',
               'VID_20210414_141014.mp4',
               'VID_20210414_141026.mp4',
               'VID_20210414_141044.mp4',
               'VID_20210414_141055.mp4',
               'VID_20210414_141111.mp4',
               'VID_20210414_141122.mp4',
               'VID_20210414_141135.mp4',
               'VID_20210414_141444.mp4',
               'VID_20210414_141452.mp4',
               'VID_20210414_141507.mp4',
               'VID_20210414_141517.mp4',
               'VID_20210414_141528.mp4',
               'VID_20210414_141541.mp4',
               'VID_20210414_141555.mp4',
               'VID_20210414_141606.mp4',
               'VID_20210414_141620.mp4',
               'VID_20210414_141633.mp4',
               'VID_20210414_142043.mp4',
               'VID_20210414_142055.mp4',
               'VID_20210414_142111.mp4',
               'VID_20210414_142126.mp4',
               'VID_20210414_142141.mp4',
               'VID_20210414_142156.mp4',
               'VID_20210414_142208.mp4',
               'VID_20210414_143311.mp4',
               'VID_20210414_143324.mp4',
               'VID_20210414_143335.mp4',
               'VID_20210414_143347.mp4',
               'VID_20210414_143413.mp4',
               'VID_20210414_143421.mp4',
               'VID_20210414_143432.mp4',
               'VID_20210414_143445.mp4',
               'VID_20210414_143458.mp4',
               'VID_20210414_143510.mp4',
               'VID_20210414_144259.mp4',
               'VID_20210414_145257.mp4',
               'VID_20210414_145307.mp4',
               'VID_20210414_145333.mp4',
               'VID_20210414_145341.mp4',
               'VID_20210414_145352.mp4',
               'VID_20210414_145509.mp4',
               'VID_20210414_145520.mp4',
               'VID_20210414_145530.mp4',
               'VID_20210414_145542.mp4',
               'VID_20210414_145615.mp4',
               'VID_20210414_145628.mp4',
               'VID_20210414_145640.mp4',
               'VID_20210414_145817.mp4',
               'VID_20210414_145824.mp4',
               'VID_20210414_145829.mp4',
               'VID_20210414_145847.mp4',
               'VID_20210414_145856.mp4',
               'VID_20210414_145905.mp4',
               'VID_20210414_150208.mp4',
               'VID_20210414_150244.mp4',
               'VID_20210414_150417.mp4',
               'VID_20210414_150430.mp4',
               'VID_20210414_150434.mp4',
               'VID_20210414_150555.mp4',
               'VID_20210414_151827.mp4',
               'VID_20210414_152114.mp4',
               'VID_20210414_153226.mp4']
stripped_names = [n.split('.')[0] for n in video_names]
hand_annotated = {'IMG_2139': slice(0, 6859),
                  'IMG_2140': slice(30 * 12, 30 * 24),
                  'IMG_2141': slice(30 * 24, 30 * 36),
                  'IMG_2142': slice(30 * 36, 30 * 48),
                  'IMG_2320': slice(30 * 48, 30 * 60),
                  'VID_20210123_091729': slice(30 * 60, 30 * 72),
                  'VID_20210123_104706': slice(30 * 72, 30 * 84),
                  'VID_20210123_110129': slice(30 * 84, 30 * 96),
                  'VID_20210123_111337': slice(30 * 96, 30 * 108),
                  'VID_20210123_111921': slice(30 * 108, 30 * 120)}


class ClimbingDataset(Dataset):
    # look at MEVA/meva/dataloaders/dataset_2d.py for inspo

    def __init__(self,
                 mode: str,
                 video_folder=os.path.expanduser('~/ucph-erda-home/videos'),
                 anno_folder=os.path.expanduser(
                     '~/ucph-erda-home/annotations'),
                 est_folder=os.path.expanduser('~/ucph-erda-home/mmpose_anno'),
                 feat_folder=os.path.expanduser(
                     '~/ucph-erda-home/hmr_features'),
                 seq_len=90,
                 overlap=0,
                 debug=''):
        super().__init__()

        self.video_folder = video_folder
        self.anno_folder = anno_folder
        self.est_folder = est_folder
        self.feat_folder = feat_folder
        self.mode = mode
        self.seq_len = seq_len
        self.overlap = overlap

        self.vids = [mmcv.VideoReader(f'{video_folder}/{n}', cache_capacity=1)
                     for n in video_names]

        self.labels = {}
        self.bboxes = {}
        self.features = {}

        # sometimes it fails to read the last frame
        all_seqs = [range(len(v) - 1) for v in self.vids]
        test_seqs = [range((i * 36) * 30, (i * 36 + 18) * 30)
                     for i in range(6)]
        val_seqs = [range((i * 36 + 18) * 30, (i * 36 + 36) * 30)
                    for i in range(6)]
        train_seqs = all_seqs.copy()
        train_seqs[0] = range(0)
        seq_switch = {'all': all_seqs,
                      'test': test_seqs,
                      'val': val_seqs,
                      'train': train_seqs
                      }
        self.frames = [np.arange(s.start, s.stop) for s in seq_switch[mode]]
        self.seqs = []
        for f in self.frames:
            if len(f) == 0:
                self.seqs.append(f)
            else:
                self.seqs.append(view_as_windows(
                    f, self.seq_len, step=self.seq_len - self.overlap))
        if mode in ['test', 'val']:
            self.seqs = [np.concatenate(self.seqs)]
        self.seq_lengths = np.array([s.shape[0] for s in self.seqs])
        self.len = sum(self.seq_lengths)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return selg.get(index)

    def get(self, index):
        vid_idx, frames = self.get_indices(index)
        name = stripped_names[vid_idx]
        vid = self.vids[vid_idx]

        if name not in self.labels:
            self.load_labels(name)
        labels = self.labels[name][frames]
        bboxes = self.bboxes[name][frames]
        if name not in self.features:
            self.load_features(name)
        features = self.features[name][frames]

        # crop and transfrom keypoints
        raw_imgs = np.array(vid[frames])
        crop_res = [image_utils.get_single_image_crop_wtrans(
            img, bbox, kps, scale=1.2) for img, bbox, kps in zip(raw_imgs.copy(), bboxes.copy(), labels.copy())]
        norm_imgs, _, kp_2d, trans, inv_trans = zip(*crop_res)
        norm_imgs, kp_2d = torch.stack(norm_imgs), np.stack(kp_2d)
        trans, inv_trans = np.stack(trans), np.stack(inv_trans)

        target = {'raw_imgs': raw_imgs,
                  'norm_imgs': norm_imgs,
                  'features': features,
                  'raw_kp_2d': labels,
                  'kp_2d': kp_2d,
                  'vid_idx': vid_idx,
                  'frames': frames,
                  'bboxes': bboxes,
                  'trans': trans,
                  'inv_trans': inv_trans}
        return target

    def get_indices(self, index):
        if self.mode in ["test", "val"]:
            vid_idx = 0
            seq_idx = index
        else:
            vid_idx = np.argmax(index < np.cumsum(self.seq_lengths))
            seq_idx = index - self.seq_lengths[:vid_idx].sum()
        seq = self.seqs[vid_idx][seq_idx]
        frames = slice(seq[0], seq[-1] + 1)
        return vid_idx, frames

    def load_labels(self, name):
        # print(f'Reading labels for {name}')
        labels = np.load(f'{self.est_folder}/{name}.npy', allow_pickle=True)

        # clip labels to image size
        vid_shape = self.vids[stripped_names.index(name)].resolution
        labels[:, :, 0] = np.clip(labels[:, :, 0], 0, vid_shape[0])
        labels[:, :, 1] = np.clip(labels[:, :, 1], 0, vid_shape[1])

        labels = kp_utils.convert_kps(labels, 'mmpose', 'spin')
        if name in hand_annotated:
            hand_labels = read_cvat_anno(
                f'{self.anno_folder}/{name}.xml')
            # add confidence of 1
            hand_labels = np.concatenate(
                (hand_labels, np.ones((hand_labels.shape[0], 19, 1))), axis=2)
            hand_labels = kp_utils.convert_kps(hand_labels, 'climb', 'spin')
            annotated_frames = hand_annotated[name]
            labels[annotated_frames] = hand_labels[annotated_frames]

        # threshold labels confidence
        threshold = 0.2
        labels[labels[:, :, -1] <= threshold] = 0

        bboxes = image_utils.get_bbox_from_kp2d(labels)

        self.labels[name] = labels
        self.bboxes[name] = bboxes

    def load_features(self, name):
        features = []
        if self.feat_folder is not None:
            # print(f'Reading features for {name}')
            feat_res = np.load(
                f'{self.feat_folder}/{name}.npy', allow_pickle=True)
            features = [r['features'] for r in feat_res]
            features = np.stack(features)

        self.features[name] = features
