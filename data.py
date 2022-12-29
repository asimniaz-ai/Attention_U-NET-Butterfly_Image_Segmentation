#It will help us build pipeline for the trianing

import os
import numpy as np
import  cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path): # we won't take image size, because we already applied it
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """Reading Image"""
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 # it will normailze image between 0 to 1
        # The (2,1,0) in transpose means the permutation to the indices.
        # For the CNN, it probably want turn a tensor of shape: [width, height, channels]
        # into: [channels, height, width]
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples