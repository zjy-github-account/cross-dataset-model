import os.path as osp
import scipy.io as sio
import numpy as np
import torch
import h5py
import scipy.signal
import scipy.io as sciio
from sklearn import preprocessing
from torch.utils.data import Dataset

class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        x_tensor = torch.from_numpy(x_tensor).float()
        y_tensor = torch.from_numpy(y_tensor).long()
        self.x = x_tensor
        self.y = y_tensor

        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)

class eegDataset2(Dataset):
    def __init__(self, data, data_4_8, data_8_12, data_12_38, label, indices):
        self.data = self.get_indices(data, indices)
        self.data_4_8= self.get_indices(data_4_8,indices)
        self.data_8_12 = self.get_indices(data_8_12, indices)
        self.data_12_38 = self.get_indices(data_12_38, indices)
        self.label = self.get_indices_label(label, indices)

    def get_indices(self, data, indices):
        data = data.numpy()
        data = data[np.array(indices),:,:,:]
        data = data.reshape(-1,1,data.shape[2],data.shape[3])
        data = torch.from_numpy(data).float()
        return data

    def get_indices_label(self, label, indices):
        label = label.numpy()
        label = label[np.array(indices), :]
        label = label.reshape(-1)
        label = torch.from_numpy(label).long()
        return label

    def __getitem__(self, index):
        return self.data[index],self.data_4_8[index],self.data_8_12[index],self.data_12_38[index], self.label[index]

    def __len__(self):
        return len(self.label)

class eegDataset3(Dataset):
    def __init__(self, data,data_4_8,data_8_12,data_12_38, label):
        self.data = self.preprocess(data)
        self.data_4_8= self.preprocess(data_4_8)
        self.data_8_12 = self.preprocess(data_8_12)
        self.data_12_38 = self.preprocess(data_12_38)
        self.label = torch.from_numpy(label).long()

    def preprocess(self, data, band=None):
        data = torch.from_numpy(data).float()
        return data

    def __getitem__(self, index):
        return self.data[index],self.data_4_8[index],self.data_8_12[index],self.data_12_38[index], self.label[index]

    def __len__(self):
        return len(self.label)

class eegDataset4(Dataset):
    def __init__(self, data, label, indices):
        # print('start')
        self.data = self.get_indices(data, indices)
        self.label = self.get_indices_label(label, indices)

    def get_indices(self, data, indices):
        data = data.numpy()
        data = data[np.array(indices),:,:,:]
        data = data.reshape(-1,1,data.shape[2],data.shape[3])
        data = torch.from_numpy(data).float()
        return data

    def get_indices_label(self, label, indices):
        label = label.numpy()
        label = label[np.array(indices), :]
        label = label.reshape(-1)
        label = torch.from_numpy(label).long()
        return label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

