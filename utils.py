import os
import glob
import torchvision
import matplotlib.pyplot as plt

def makedirs_ifno(paths):
    for path in paths:
        if os.path.exists(path):
            files = glob.glob(f'{path}*')
            for f in files:
                os.remove(f)
        else:
            os.makedirs(path)
            
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = torchvision.transforms.functional.to_pil_image(inp)
    plt.imshow(inp)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated