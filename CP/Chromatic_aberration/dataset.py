
import numpy as np
from PIL import Image
from skimage import transform
from skimage import img_as_float32
from torch.utils.data import Dataset




class ChromaticAberrationDataset(Dataset):
  def __init__(self, patch_dim, gap, df, validate, transforms=None):
    self.patch_dim, self.gap = patch_dim, gap
    self.transform = transforms

    if validate:
      self.train_data = df.values
    else:
      self.train_data = df.values

  
  def get_patches_and_coordinates(self, image, patch_dim, gap):
    patch_loc_arr = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    patch_coordinates = []
    offset_x = image.shape[0] - (patch_dim*3 + gap*2)
    offset_y = image.shape[1] - (patch_dim*3 + gap*2)

    start_grid_x, start_grid_y = 9, 9

    patch_bucket = np.empty([9, 3, 96, 96], dtype='float32')

    for i, (tempx, tempy) in enumerate(patch_loc_arr):
        tempx, tempy = patch_loc_arr[i]

        patch_x_pt = start_grid_x + patch_dim * (tempx-1) + gap * (tempx-1)
        patch_y_pt = start_grid_y + patch_dim * (tempy-1) + gap * (tempy-1)

        patch_coordinates.append([patch_x_pt, patch_y_pt])
        img_patch = image[patch_x_pt:patch_x_pt+patch_dim, patch_y_pt:patch_y_pt+patch_dim]

        # Resizing the patch to 96x96
        if img_patch.shape[0] != 96:
          img_patch = transform.resize(img_patch, (96, 96)) 
          img_patch = img_as_float32(img_patch)

        patch_bucket[i] = np.transpose(img_patch, (2, 0, 1))

    return patch_bucket, np.array(patch_coordinates)

  def __len__(self):
    return len(self.train_data)
  
  def __getitem__(self, index):
    image = np.array(Image.open(self.train_data[index]).convert('RGB'))
    patch_bucket, coordinates = self.get_patches_and_coordinates(image, self.patch_dim, self.gap)

    coordinates = coordinates.astype(np.float32)

    return patch_bucket, coordinates 
