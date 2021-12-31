import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')


def str2bool(v):
    """ For arguments parser """
    return v.lower() in ("yes", "true", "t", "1")


class UnNormalize(object):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
    Returns:
        Tensor: Normalized image.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def convert_format(data, format):
  if format == 'p':
    return np.transpose(data, (0, 3, 1, 2))
  if format == 'n':
    return np.transpose(data, (0, 2, 3, 1))
  if format == '3':
    return np.transpose(data, (1, 2, 0))


def imshow(img,text=None,should_save=False):
    plt.figure(figsize=(10, 10))
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  

def show_plot(iteration,loss,fname):
    plt.plot(iteration,loss)
    plt.savefig(fname)
    plt.show()

# Plotting losses
def plot_losses(ckpt_path = ''):
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    plt.plot(range(len(checkpoint['global_trnloss'])), checkpoint['global_trnloss'], 
             label='TRAIN Loss')
    plt.plot(range(len(checkpoint['global_valloss'])), checkpoint['global_valloss'], 
             label='VAL Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Main Network Training/Validation Loss plot')
    plt.legend()
    plt.show()
