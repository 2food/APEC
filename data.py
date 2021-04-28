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

  for i, track in enumerate(root.findall('track')):
    for j, point in enumerate(track.findall('points')):
      p_strs = point.attrib['points'].split(',')
      data[j, i] = list(map(float, p_strs))
  return data


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


class ClimbingDataset(Dataset):
  # look at MEVA/meva/dataloaders/dataset_2d.py for inspo

  video_names = ['IMG_2139.MOV',
                 'IMG_2140.MOV',
                 'IMG_2141.MOV',
                 'IMG_2142.MOV'
                 #'IMG_2320.mov',
                 #'VID_20210123_091729.mp4',
                 #'VID_20210123_104706.mp4',
                 #'VID_20210123_110129.mp4',
                 #'VID_20210123_111337.mp4',
                 #'VID_20210123_111921.mp4'
                 ]

  stripped_names = [n.split('.')[0] for n in video_names]
  test_seqs = [slice((i * 12) * 30, (i * 12 + 6) * 30)
               for i, _ in enumerate(video_names)]
  val_seqs = [slice((i * 12 + 6) * 30, (i * 12 + 12) * 30)
              for i, _ in enumerate(video_names)]

  def __init__(self,
               video_folder: str,
               anno_folder: str,
               mode: str,
               feat_folder=None,
               seq_len=90,
               overlap=0):
    super().__init__()
    self.vids = [mmcv.VideoReader(f'{video_folder}/{n}')
                 for n in self.video_names]
    self.labels = [read_cvat_anno(f'{anno_folder}/{n}.xml')
                   for n in self.stripped_names]

    self.features = []
    if feat_folder is not None:
      print(f'Reading features for ...', end=' ')
      for i, vid_name in enumerate(self.stripped_names):
        features = []
        folder = f'{feat_folder}/{vid_name}'
        print(vid_name, end=' ')
        for file in os.listdir(f'{feat_folder}/{vid_name}'):
          res = np.load(f'{folder}/{file}', allow_pickle=True)
          features.append(res.item()['features'])
        self.features.append(np.stack(features))

    self.bboxes = [image_utils.get_bbox_from_kp2d(
        l).T for l in self.labels]
    self.seq_len = seq_len
    self.overlap = overlap

    self.all_seqs = [slice(0, min(len(v), 120 * 30))
                     for v in self.vids]  # annotated 2 mins
    self.all_seqs[0] = slice(0, len(self.vids[0]))  # fully annotated

    seq_switch = {'all': self.all_seqs,
                  'test': self.test_seqs,
                  'val': self.val_seqs}
    self.frames = [np.arange(s.start, s.stop) for s in seq_switch[mode]]
    self.seqs = [view_as_windows(
        f, self.seq_len, step=self.seq_len - self.overlap) for f in self.frames]
    self.seq_lengths = np.array([s.shape[0] for s in self.seqs])
    self.len = sum(self.seq_lengths)

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    """Gets a sequence of SMPL features."""
    vid_idx, frames = self.get_indices(index)
    return self.features[vid_idx][frames]

  def get(self, index):
    """Gets more info about the sequence than __getitem__."""
    vid_idx, frames = self.get_indices(index)

    vid = self.vids[vid_idx]

    bboxes = self.bboxes[vid_idx][frames]
    labels = self.labels[vid_idx][frames]
    # add confidence of 1
    labels = np.concatenate(
        (labels, np.ones((labels.shape[0], 19, 1))), axis=2)

    # crop and transfrom keypoints
    raw_imgs = np.array(vid[frames])
    crop_res = [image_utils.get_single_image_crop_wtrans(
        img, bbox, kps, scale=1.2) for img, bbox, kps in zip(raw_imgs.copy(), bboxes.copy(), labels.copy())]
    norm_imgs, _, kp_2d, trans, inv_trans = zip(*crop_res)
    norm_imgs, kp_2d = torch.stack(norm_imgs), np.stack(kp_2d)
    trans, inv_trans = np.stack(trans), np.stack(inv_trans)

    # convert keypoints to spin format
    kp_utils.get_climb_joint_names = get_climb_joint_names
    kp_2d = kp_utils.convert_kps(kp_2d, 'climb', 'spin')

    features = []
    if len(self.features) > 0:
      features = self.features[vid_idx][frames]

    target = {'raw_imgs': raw_imgs,
              'norm_imgs': norm_imgs,
              'features': features,
              'climb_labels': labels,
              'kp_2d': kp_2d,
              'vid_idx': vid_idx,
              'frames': frames,
              'bboxes': bboxes,
              'trans': trans,
              'inv_trans': inv_trans}
    return target

  def get_indices(self, index):
    vid_idx = np.argmax(index < np.cumsum(self.seq_lengths))
    seq_idx = index - self.seq_lengths[:vid_idx].sum()
    seq = self.seqs[vid_idx][seq_idx]
    frames = slice(seq[0], seq[-1] + 1)
    return vid_idx, frames
