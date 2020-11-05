from torch.utils.data import Dataset
from PIL import Image
import torch
import config
import torchvision.transforms as transforms
import numpy as np
from sklearn import preprocessing

def readTxt(file_path):
    """
    Reads a list from a list.

    Args:
        file_path: (str): write your description
    """
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    return img_list

class RoadSequenceDataset(Dataset):

    def __init__(self, file_path, transforms):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            file_path: (str): write your description
            transforms: (str): write your description
        """

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        """
        Returns the number of rows in the dataset.

        Args:
            self: (todo): write your description
        """
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Retrieve image for the given index.

        Args:
            self: (todo): write your description
            idx: (list): write your description
        """
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[4])
        label = Image.open(img_path_list[5])
        data = self.transforms(data)
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample

class RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, transforms):
        """
        Initialize the dataset.

        Args:
            self: (todo): write your description
            file_path: (str): write your description
            transforms: (str): write your description
        """

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        """
        Returns the number of rows in the dataset.

        Args:
            self: (todo): write your description
        """
        return self.dataset_size

    def __getitem__(self, idx):
        """
        Get the image from index.

        Args:
            self: (todo): write your description
            idx: (list): write your description
        """
        img_path_list = self.img_list[idx]
        data = []
        for i in range(5):
            data.append(torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0))
        data = torch.cat(data, 0)
        label = Image.open(img_path_list[5])
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample


