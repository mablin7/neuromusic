# import numpy as np
import torch
from torch.utils import data
import h5py
import numpy as np

INPUT_SPACE = 130

class MelodySequenceDataset(data.Dataset):
    def __init__(self, hdf_file, subset=None, in_memory=False):
        super(MelodySequenceDataset, self).__init__()
        # self.npz = np.load(npz_file)
        if in_memory:
            print('Loading dataset, this will take a while...')
        self.hdf = h5py.File(hdf_file, 'r', libver='latest', driver=('core' if in_memory else None))
        self.files = list(self.hdf['data'].keys()) #self.npz.files
        self.subset = subset
    
    def __len__(self):
        return len(self.files) if self.subset is None else min(self.subset, len(self.files))
    
    def __getitem__(self, index):
        # return torch.as_tensor(self.npz[self.files[index]])
        data = self.hdf['data'][self.files[index]]
        arr = np.empty(data.shape, dtype=np.int64)
        data.read_direct(arr)
        np.save('latest.npy', arr)
        return torch.as_tensor(arr) 
    