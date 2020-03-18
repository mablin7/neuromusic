# import numpy as np
import torch
from torch.utils import data
import h5py

INPUT_SPACE = 130

class MelodySequenceDataset(data.Dataset):
    def __init__(self, hdf_file, subset=None):
        super(MelodySequenceDataset, self).__init__()
        # self.npz = np.load(npz_file)
        self.hdf = h5py.File(hdf_file, 'r', libver='latest', swmr=True)
        self.files = list(self.hdf['data'].keys()) #self.npz.files
        self.subset = subset
    
    def __len__(self):
        return len(self.files) if self.subset is None else min(self.subset, len(self.files))
    
    def __getitem__(self, index):
        # return torch.as_tensor(self.npz[self.files[index]])
        return torch.as_tensor(self.hdf['data'][self.files[index]])
            
def get_data(hdf_file, subset=None, train_ratio=0.8):
    dataset = MelodySequenceDataset(hdf_file, subset)
    dl = len(dataset)
    train_l = int(train_ratio * dl)
    test_l = dl - train_l
    train_set, test_set = data.random_split(dataset, [train_l, test_l])
    train_loader = data.DataLoader(train_set, batch_size=None, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=None, num_workers=4)
    return train_loader, test_loader