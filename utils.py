import os
import glob

def makedirs_ifno(paths):
    for path in paths:
        if os.path.exists(path):
            files = glob.glob(f'{path}*')
            for f in files:
                os.remove(f)
        else:
            os.makedirs(path)