import sys
import os
sys.path.append(os.getcwd())
from meva.lib.spin import get_pretrained_hmr
from tqdm import tqdm
import argparse
import data
import numpy as np
from utils import makedirs_ifno

def main(args):

    vid_folder = args.vid_folder
    anno_folder = args.anno_folder
    out_folder = args.out_folder

    # read dataset with seq_len=1 to ensure that all frames have features
    c = data.ClimbingDataset(vid_folder, anno_folder,
                             'all', seq_len=1)

    hmr = get_pretrained_hmr()
    hmr.eval()

    makedirs_ifno([f'{out_folder}/{n}/' for n in c.stripped_names])

    for seq in tqdm(c):
        imgs = seq['norm_imgs']
        vid_name = c.stripped_names[seq['vid_idx']]
        frames = seq['frames']
        for f, img in zip(frames, imgs):
            img = img.float().unsqueeze(0).to('cuda')
            feat, out = hmr(img, return_features=True)
            feat = feat.cpu().detach().numpy()[0]
            out = out[0]
            out = dict([(k, out[k].cpu().detach().numpy()) for k in out])
            out['features'] = feat
            np.save(f'{out_folder}/{vid_name}/{f:06d}', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_folder', type=str, help='input video folder')
    parser.add_argument('--anno_folder', type=str,
                        help='input annotations folder')
    parser.add_argument('--out_folder', type=str, help='feature output folder')

    args = parser.parse_args()

    main(args)
