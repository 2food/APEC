from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torch.utils.data import Dataset
from typing import Any, Callable, Optional
import mmcv
from mmcv import VideoReader

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

        
class ClimbingDataset(Dataset):
    # look at MEVA/meva/dataloaders/dataset_2d.py for inspo
    
    video_names = ['IMG_2139.MOV', 
                   'IMG_2140.MOV',
                   'IMG_2141.MOV',
                   'IMG_2142.MOV',
                   'IMG_2320.mov',
                   'VID_20210123_091729.mp4',
                   'VID_20210123_104706.mp4',
                   'VID_20210123_110129.mp4',
                   'VID_20210123_111337.mp4',
                   'VID_20210123_111921.mp4']
    test_seqs = dict([(v,slice((i*12)*30, (i*12+6)*30)) for i,v in enumerate(video_names)])
    val_seqs = dict([(v,slice((i*12+6)*30, (i*12+12)*30)) for i,v in enumerate(video_names)])
    
    def __init(self, 
               folder: str):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass