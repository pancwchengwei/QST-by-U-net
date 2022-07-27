import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import my_utils as utils

class MyDataset(data.Dataset):
    def __init__(self, root, image_size=16, mode='train', augmentation_prob=0.):
        """Initializes image paths and preprocessing module."""
        if mode=='train':
            train_inputs, train_outputs = utils.load_data_wrapper_train(root)
            self.input = train_inputs
            self.output = train_outputs
        elif mode=='valid':
            train_inputs_val, train_outputs_val = utils.load_data_wrapper_val(root)
            self.input = train_inputs_val
            self.output = train_outputs_val
        else:
            train_inputs_test, train_outputs_test = utils.load_data_wrapper_test(root)
            # train_outputs_test = np.squeeze(train_outputs_test, 1)

            mask_test = utils.generate_mask4(train_inputs_test, 16)
            mask_test = np.expand_dims(mask_test, 1)
            train_inputs_test = train_inputs_test * mask_test
            index = 400
            # mask_test = mask_test[index:index+100]
            train_inputs_test = train_inputs_test[index:index+100]
            train_outputs_test = train_outputs_test[index:index+100]


            self.input = train_inputs_test
            self.output = train_outputs_test


    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        return self.input[index], self.output[index]

    def __len__(self):
        """Returns the total number of font files."""
        return self.input.shape[0]


def get_loader(image_path, image_size, batch_size, num_workers=0, mode='train', augmentation_prob=0.):
    """Builds and returns Dataloader."""

    dataset = MyDataset(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
