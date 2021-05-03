# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import sys
import os
sys.path.append(os.getcwd())
import joblib
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm

from meva.utils.video_config import AMASS_DIR

dict_keys = ['betas', 'dmpls', 'gender', 'mocap_framerate', 'poses', 'trans']

# extract SMPL joints from SMPL-H model
joints_to_use = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 37
])
joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)

all_sequences = [
    'ACCAD',
    #'BioMotionLab_NTroje',
    'BMLhandball'
    #'BMLmovi',
    #'CMU',
    #'DFaust_67',
    #'EKUT',
    #'Eyes_Japan_Dataset',
    #'HumanEva',
    #'KIT',
    #'MPI_HDM05',
    #'MPI_Limits',
    #'MPI_mosh',
    #'SFU',
    #'SSM_synced',
    #'TCD_handMocap',
    #'TotalCapture',
    #'Transitions_mocap',
]


def read_data(folder, sequences):
    # sequences = [osp.join(folder, x) for x in sorted(os.listdir(folder)) if osp.isdir(osp.join(folder, x))]

    if sequences == 'all':
        sequences = all_sequences

    db = {}

    for seq_name in sequences:
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        seq = read_single_sequence(
            seq_folder, seq_name)

        poses = seq['poses']
        seq_name_list = np.array([seq_name] * poses.shape[0])
        print(seq_name, 'number of videos', poses.shape[0])
        db[seq_name] = seq

    return db


def read_single_sequence(folder, seq_name):
    subjects = os.listdir(folder)

    seq = {
        'poses': [],
        'trans': [],
        'mocap_framerate': [],
        'betas': [],
        'dmpls': [],
        'vid_name': [],
    }

    for subject in tqdm(subjects):
        actions = [x for x in os.listdir(
            osp.join(folder, subject)) if x.endswith('.npz')]

        for action in actions:
            fname = osp.join(folder, subject, action)

            if fname.endswith('shape.npz'):
                continue

            data = np.load(fname)

            poses = data['poses'][:, joints_to_use]

            if poses.shape[0] < 90:
                continue
            trans = data['trans']
            mocap_framerate = data['mocap_framerate'].item()
            betas = data['betas']
            dmpls = data['dmpls']
            vid_name = np.array(
                [f'{seq_name}_{subject}_{action[:-4]}'] * poses.shape[0])

            seq['poses'].append(poses)
            seq['trans'].append(trans)
            seq['mocap_framerate'].append(mocap_framerate)
            seq['betas'].append(betas)
            seq['dmpls'].append(dmpls)
            seq['vid_name'].append(vid_name)

    seq['poses'] = np.concatenate(seq['poses'], axis=0)
    seq['trans'] = np.concatenate(seq['trans'], axis=0)
    seq['mocap_framerate'] = seq['mocap_framerate'][0]
    seq['betas'] = np.concatenate(seq['betas'], axis=0)
    seq['dmpls'] = np.concatenate(seq['dmpls'], axis=0)
    seq['vid_name'] = np.concatenate(seq['vid_name'], axis=0)

    return seq


def read_seq_data(folder, nsubjects, fps):
    subjects = os.listdir(folder)
    sequences = {}

    assert nsubjects < len(
        subjects), 'nsubjects should be less than len(subjects)'

    for subject in subjects[:nsubjects]:
        actions = os.listdir(osp.join(folder, subject))

        for action in actions:
            data = np.load(osp.join(folder, subject, action))
            mocap_framerate = int(data['mocap_framerate'])
            sampling_freq = mocap_framerate // fps
            sequences[(subject, action)
                      ] = data['poses'][0:: sampling_freq, joints_to_use]

    train_set = {}
    test_set = {}

    for i, (k, v) in enumerate(sequences.items()):
        if i < len(sequences.keys()) - len(sequences.keys()) // 4:
            train_set[k] = v
        else:
            test_set[k] = v

    return train_set, test_set


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        help='dataset directory', default='data/amass')
    args = parser.parse_args()

    db = read_data(args.dir, sequences=all_sequences)
    db_file = osp.join(AMASS_DIR, 'amass_db.pt')
    print(f'Saving AMASS dataset to {db_file}')
    joblib.dump(db, db_file)
