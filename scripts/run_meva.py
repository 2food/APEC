import sys
import os
sys.path.append(os.getcwd())
from meva.lib.meva_model import MEVA
from meva.utils.video_config import update_cfg
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import data
import joblib
import numpy as np
import os
import torch
import time
from tqdm import tqdm
from mesh import render_vids
import utils


def main(args):
    out_folder = args.out_folder
    cfg = args.cfg

    utils.makedirs_ifno([out_folder])

    print('Loading climbing data ...')
    c = data.ClimbingDataset('all', seq_len=90, preload_all=True)
    print('Done')

    # load pretrained MEVA
    print('Loading MEVA model ...')
    pretrained_file = f"results/meva/{cfg}/model_best.pth.tar"
    config_file = f"meva/cfg/{cfg}.yml"
    cfg = update_cfg(config_file)

    device = torch.device(cfg.DEVICE)
    batch_size = cfg.TRAIN.BATCH_SIZE

    model = MEVA(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=batch_size,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
        cfg=cfg.VAE_CFG,
    ).to(device)
    ckpt = torch.load(pretrained_file, map_location=device)
    # print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print('Done')

    dataloader = DataLoader(
        c, batch_size=batch_size, num_workers=6, shuffle=False)

    with torch.no_grad():
        pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [
        ], [], [], [], [], []
        vid_indices = []
        start = time.time()
        for target in tqdm(iter(dataloader)):

            feats = target['features'].to(device)
            output = model(feats)[-1]

            theta = output['theta'].cpu()
            pred_cam.append(theta[:, :, :3])
            pred_verts.append(output['verts'].cpu())
            pred_pose.append(theta[:, :, 3:75])
            pred_betas.append(theta[:, :, 75:])
            pred_joints3d.append(output['kp_3d'].cpu())
            norm_joints2d.append(output['kp_2d'].cpu())
            vid_indices.append(target['vid_idx'])
        finish = time.time()
        total_time = finish - start
        pred_cam = torch.cat(pred_cam, dim=0)
        pred_verts = torch.cat(pred_verts, dim=0)
        pred_pose = torch.cat(pred_pose, dim=0)
        pred_betas = torch.cat(pred_betas, dim=0)
        pred_joints3d = torch.cat(pred_joints3d, dim=0)
        norm_joints2d = torch.cat(norm_joints2d, dim=0)
        vid_indices = torch.cat(vid_indices, dim=0)

    # ========= Save results to a pickle file ========= #
    pred_cam = pred_cam.numpy()
    pred_verts = pred_verts.numpy()
    pred_pose = pred_pose.numpy()
    pred_betas = pred_betas.numpy()
    pred_joints3d = pred_joints3d.numpy()
    norm_joints2d = norm_joints2d.numpy()
    vid_indices = vid_indices.numpy()

    print('Saving results...')
    output_dict = {'total_time': total_time}
    for vid_idx in tqdm(np.unique(vid_indices)):
        vid_name = data.stripped_names[vid_idx]
        inds = vid_indices == vid_idx

        output_dict = {'pred_cam': pred_cam[inds],
                       'verts': pred_verts[inds],
                       'pose': pred_pose[inds],
                       'betas': pred_betas[inds],
                       'joints3d': pred_joints3d[inds],
                       'joints2d': norm_joints2d[inds]}

        joblib.dump(output_dict, os.path.join(out_folder, f"{vid_name}.pkl"))

    if args.render_vids:
        print('Rendering videos...')
        render_vids(c, output_dict, out_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, help='config file')
    parser.add_argument('--out_folder', type=str, help='output folder')
    parser.add_argument('--render_vids', action='store_true',
                        help='whether to render mesh videos')

    args = parser.parse_args()

    main(args)
