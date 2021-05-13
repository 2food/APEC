import data


def Climb(split='train', seqlen=90, overlap=0.75, debug=False):
    modes = {'train': 'train',
             'test': 'val'}
    overlap = int(overlap * seqlen)
    return data.ClimbingDataset(modes[split], seq_len=seqlen, overlap=overlap, preload_all=True)
