from functools import partial
import torch
import os
import PIL
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __setup_files(self):

        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, data_path, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        # define the state of the object
        self.data_dir = data_path
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()

    
    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        # read the image:
        img_name = self.files[idx]
        if img_name[-4:] == ".npy":
            img = np.load(img_name)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))
        else:
            img = Image.open(img_name)

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        # return the image:
        return img    