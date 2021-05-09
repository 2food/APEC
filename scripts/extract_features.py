import sys
import os
sys.path.append(os.getcwd())
from meva.lib.spin import get_pretrained_hmr
from tqdm.auto import tqdm
import argparse
import data
import numpy as np
from utils import makedirs_ifno


def main(args):
    out_folder = args.out_folder

    # read dataset with seq_len=1 to ensure that all frames have features
    c = data.ClimbingDataset('all', feat_folder=None, seq_len=1)

    hmr = get_pretrained_hmr()
    hmr.eval()

    makedirs_ifno([f'{out_folder}'])

    current_vid_name = data.stripped_names[0]
    outs = []
    c_range = range(len(c))
    for i in tqdm(c_range):
        seq = c.get(i)
        imgs = seq['norm_imgs']
        vid_name = data.stripped_names[seq['vid_idx']]
        frames = seq['frames']
        frames = range(frames.start, frames.stop)
        for f, img in zip(frames, imgs):
            img = img.float().unsqueeze(0).to('cuda')
            feat, out = hmr(img, return_features=True)
            feat = feat[0].cpu().detach().numpy()
            out = out[0]
            out = dict([(k, out[k].cpu().detach().numpy()) for k in out])
            out['features'] = feat
            outs.append(out)
        if vid_name != current_vid_name:
            np.save(f'{out_folder}/{current_vid_name}.npy', outs)
            outs = []
            current_vid_name = vid_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_folder', type=str, help='feature output folder')

    args = parser.parse_args()

    main(args)
