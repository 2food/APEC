import sys
import os
sys.path.append(os.getcwd())
from meva.lib.spin import get_pretrained_hmr
import mmcv
import numpy as np
import tqdm
import argparse

def main(args):

    vid_folder = args.vid_folder
    vid_name = args.vid_name
    out_folder = args.out_folder

    stripped_name = vid_name.split('.')[0]
    vid_file = f'{vid_folder}/{vid_name}'
    vid = mmcv.video.VideoReader(vid_file)

    hmr = get_pretrained_hmr()
    hmr.eval()

    for img in tqdm(vid):
        img = image_utils.convert_cvimg_to_tensor(img)
        img = img.unsqueeze(0).float().to('cuda')
        feat = hmr.feature_extractor(img).to('cpu')
        np.save(f'{out_folder}/{stripped_name}/{i:06d}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_folder', type=str, help='input video folder')
    parser.add_argument('--vid_name', type=str, help='input video file name')
    parser.add_argument('--out_folder', type=str, help='output folder')

    args = parser.parse_args()

    main(args)
