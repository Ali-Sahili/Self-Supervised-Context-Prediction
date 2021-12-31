
import torch.nn as nn


class ColorAbberationNetwork(nn.Module):
  def __init__(self):
    super(ColorAbberationNetwork, self).__init__() 
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 96, kernel_size=11, stride=4),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.LocalResponseNorm(96),
      
      nn.Conv2d(96, 384, kernel_size=5, stride = 2,padding = 2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.LocalResponseNorm(384),
      
      nn.Conv2d(384, 384, kernel_size=3, stride=1,padding = 1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(384),
      
      nn.Conv2d(384, 384, kernel_size=3, stride=1,padding = 1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(384),
      
      nn.Conv2d(384, 256, kernel_size=3, stride=1,padding = 1),
      nn.ReLU(inplace=True),
      nn.BatchNorm2d(256),
      nn.MaxPool2d(kernel_size=3, stride=2,padding = 1),
    )
    self.fc6 = nn.Sequential(
      nn.Linear(256,4096),
      nn.ReLU(inplace=True),
      nn.BatchNorm1d(4096),
    )
    self.fc = nn.Sequential(
      nn.Linear(4096, 2)
    )

  def forward(self, patch_bunch):
    bs, _, _, _ = patch_bunch.shape
    output = self.cnn(patch_bunch)
    output = output.view(bs, -1)
    output = self.fc6(output)
    output = self.fc(output)
    return output
