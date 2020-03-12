import numpy as np
import torch
from torch.utils import data

INPUT_SPACE = 130

class MelodySequenceDataset(data.Dataset):
    def __init__(self, npz_file):
        super(MelodySequenceDataset, self).__init__()
        self.npz = np.load(npz_file)
        self.files = self.npz.files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        dat = torch.from_numpy(self.npz[self.files[index]]).view(-1, 1)
        onehot = torch.zeros(dat.shape[0], INPUT_SPACE)
        onehot = onehot.scatter_(1, dat, 1)
        onehot = onehot.view(1, -1, INPUT_SPACE)
        return onehot
    
    # def _create_batches_from_list(self, num_batches, l):
    #     batches = [[] for _ in range(num_batches)]
    #     batch_iter = itertools.cycle(batches)
    #     for f in l:
    #         next_batch = next(batch_iter)
    #         next_batch.append(f)
    #     return batches

    # def _melody_generator(self, files):
    #     for f in files:
    #         yield self.npz[f]  

    # def __iter__(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:
    #         return _melody_generator(self.npz, 0)
    #     else:
    #         worker_batches = self._create_batches_from_list(worker_info.num_workers, self.files)
    #         curr_batch = worker_batches[worker_info.id]
    #         return self._melody_generator(curr_batch)
            
def get_data(npz_file, train_ratio=0.8):
    dataset = MelodySequenceDataset(npz_file)
    dl = len(dataset)
    train_l = int(train_ratio * dl)
    test_l = dl - train_l
    train_set, test_set = data.random_split(dataset, [train_l, test_l])
    return train_set, test_set