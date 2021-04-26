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


def main(args):
    device = 'cuda'
    vid_folder = args.vid_folder
    anno_folder = args.anno_folder
    feat_folder = args.feat_folder
    out_folder = args.out_folder
    c = data.ClimbingDataset(vid_folder, anno_folder,
                             'all', seq_len=90, feat_folder=feat_folder)

    # load pretrained MEVA
    pretrained_file = f"results/meva/train_meva_2/model_best.pth.tar"
    config_file = f"meva/cfg/train_meva_2.yml"
    cfg = update_cfg(config_file)
    model = MEVA(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
        cfg=cfg.VAE_CFG,
    ).to(device)
    ckpt = torch.load(pretrained_file)
    # print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt)
    model.eval()

    dataloader = DataLoader(
        c, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=16, shuffle=False)

    with torch.no_grad():
        pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [
        ], [], [], [], [], []
        for seqs in dataloader.batch_sampler:
            feats = np.stack([c[seq] for seq in seqs]).to(device)
            output = model(feats)[-1]

            pred_cam.append(output['theta'][:, :, :3])
            pred_verts.append(output['verts'])
            pred_pose.append(output['theta'][:, :, 3:75])
            pred_betas.append(output['theta'][:, :, 75:])
            pred_joints3d.append(output['kp_3d'])
            norm_joints2d.append(output['kp_2d'])

        pred_cam = torch.cat(pred_cam, dim=0)
        pred_verts = torch.cat(pred_verts, dim=0)
        pred_pose = torch.cat(pred_pose, dim=0)
        pred_betas = torch.cat(pred_betas, dim=0)
        pred_joints3d = torch.cat(pred_joints3d, dim=0)
        norm_joints2d = torch.cat(norm_joints2d, dim=0)

    # ========= Save results to a pickle file ========= #
    pred_cam = pred_cam.cpu().numpy()
    pred_verts = pred_verts.cpu().numpy()
    pred_pose = pred_pose.cpu().numpy()
    pred_betas = pred_betas.cpu().numpy()
    pred_joints3d = pred_joints3d.cpu().numpy()
    norm_joints2d = norm_joints2d.cpu().numpy()

    output_dict = {'pred_cam': pred_cam,
                   'verts': pred_verts,
                   'pose': pred_pose,
                   'betas': pred_betas,
                   'joints3d': pred_joints3d,
                   'joints2d': norm_joints2d}
    joblib.dump(output_dict, os.path.join(out_folder, "meva_output.pkl"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_folder', type=str, help='input video folder')
    parser.add_argument('--anno_folder', type=str,
                        help='input annotations folder')
    parser.add_argument('--feat_folder', type=str,
                        help='input feature folder')
    parser.add_argument('--out_folder', type=str, help='output folder')

    args = parser.parse_args()

    main(args)
