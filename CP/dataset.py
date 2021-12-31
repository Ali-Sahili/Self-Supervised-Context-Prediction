import os
import random
import skimage
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob

from skimage import transform
from skimage import img_as_float32
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit



# Creating training and validation datasets
def build_dataset(args, mode='train'):
    PATH = os.path.join(args.root, mode)
    df_list = [] 
    classes = os.listdir(PATH)
    
    for idx, each_cls in enumerate(classes):
        images_in_each_class = glob(f'{PATH}/{each_cls}/*.JPEG')
        df_list += [[each_image, each_cls] for each_image in images_in_each_class]

    df = pd.DataFrame(data=df_list, columns=['filename', 'class'])

    # Taking the classes subset
    num_training_classes_subset = 10
    train_classes_used = df['class'].unique()[:num_training_classes_subset]
    df = df[df['class'].isin(train_classes_used)]
    # df.groupby('class').count()

    X, y = df['filename'], df['class']
    sss = StratifiedShuffleSplit(n_splits=5, train_size=args.ratio, random_state=0)
    sss.get_n_splits(X, y)
    #print(sss)
    
    for train_index, test_index in sss.split(X, y):
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        stratified1000trn = train_index
        break

    df_trn = df.iloc[stratified1000trn].reset_index(drop=True)
    df_trn.head()

    return df_trn


# This class generates patches for training
class MyDataset(Dataset):
  def __init__(self, patch_dim, gap, df, validate, transforms=None):
    self.patch_dim, self.gap = patch_dim, gap
    self.transform = transforms

    if validate:
      self.train_data = df.values
    else:
      self.train_data = df.values
  
  def get_patch_from_grid(self, image, patch_dim, gap):
    image = np.array(image)

    offset_x, offset_y = image.shape[0]-(patch_dim*3+gap*2),image.shape[1]-(patch_dim*3 + gap*2)
    start_grid_x, start_grid_y = np.random.randint(0, offset_x), np.random.randint(0, offset_y)
    patch_loc_arr = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
    loc = np.random.randint(len(patch_loc_arr))
    tempx, tempy = patch_loc_arr[loc]
    
    patch_x_pt = start_grid_x + patch_dim * (tempx-1) + gap * (tempx-1)
    patch_y_pt = start_grid_y + patch_dim * (tempy-1) + gap * (tempy-1)
    random_patch = image[patch_x_pt:patch_x_pt+patch_dim, patch_y_pt:patch_y_pt+patch_dim]

    patch_x_pt = start_grid_x + patch_dim * (2-1) + gap * (2-1)
    patch_y_pt = start_grid_y + patch_dim * (2-1) + gap * (2-1)
    uniform_patch = image[patch_x_pt:patch_x_pt+patch_dim, patch_y_pt:patch_y_pt+patch_dim]
    
    random_patch_label = loc
    
    return uniform_patch, random_patch, random_patch_label

  def __len__(self):
    return len(self.train_data)
  
  def __getitem__(self, index):
    image = Image.open(self.train_data[index]).convert('RGB')
    uniform_patch, random_patch, random_patch_label = self.get_patch_from_grid(image, 
                                                                               self.patch_dim, 
                                                                               self.gap)
    if uniform_patch.shape[0] != 96:
        uniform_patch = transform.resize(uniform_patch, (96, 96))
        random_patch = transform.resize(random_patch, (96, 96))
        
        uniform_patch = img_as_float32(uniform_patch)
        random_patch = img_as_float32(random_patch)

    # Dropped color channels 2 and 3 and 
    # replaced with gaussian noise(std ~1/100 of the std of the remaining channel)
    uniform_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]), 
                                              (uniform_patch.shape[0],uniform_patch.shape[1]))
    uniform_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]), 
                                              (uniform_patch.shape[0],uniform_patch.shape[1]))
    random_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]), 
                                              (random_patch.shape[0],random_patch.shape[1]))
    random_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]), 
                                              (random_patch.shape[0],random_patch.shape[1]))

    random_patch_label = np.array(random_patch_label).astype(np.int64)
        
    if self.transform:
      uniform_patch = self.transform(uniform_patch)
      random_patch = self.transform(random_patch)

    return uniform_patch, random_patch, random_patch_label
