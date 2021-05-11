import data


def Climb(split='train', seqlen=90, debug=False):
    modss = {'train': 'train',
             'test': 'val'}
    return data.ClimbingDataset(modes[split], seq_len=seqlen)
