import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

class ImageDataset(Dataset):
    def __init__(self, image_data, transform=None):
        self.image_data1 = image_data['image1']
        self.image_data2 = image_data['image2']
        self.transform = transform

    def __len__(self):
        return len(self.image_data1)

    def _normalization(self, data):
        return data/255
        
    def __getitem__(self, idx):
        input1 = Image.open(self.image_data1[idx]).convert("RGB")
        input2 = Image.open(self.image_data2[idx]).convert("RGB")
        input1 = np.array(input1)
        input2 = np.array(input2)
        if self.transform:
            trasformed = self.transform(image=input1, image1=input2)
            return trasformed['image'], trasformed['image1']
        else:
            input1 = self._normalization(torch.from_numpy(input1).permute(2, 0, 1))
            input2 = self._normalization(torch.from_numpy(input2).permute(2, 0, 1))
            return torch.FloatTensor(input1), torch.FloatTensor(input2)
