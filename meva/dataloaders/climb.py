import data


def Climb(split='train', seqlen=90, debug=False):
    modes = {'train': 'train',
             'test': 'val'}
    return data.ClimbingDataset(modes[split], seq_len=seqlen)
