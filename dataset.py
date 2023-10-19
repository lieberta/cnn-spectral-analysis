import torch
from torch.utils.data import Dataset
import numpy as np
from preprocess.import_normalise import import_all_rooms
class Dataset_x1_y1(Dataset):
    def __init__(self):
        #load npz file
        #_,_,xy= npz_to_matrixlist('./Simulationdata/Sample Case Room - one mesh size/temperature_matrices.npz')

        # if true => derivation of the data is calculated and used, if false the original data is used
        xy = import_all_rooms(False)
        self.x = []
        self.y = []
        for key, _ in xy.items():
            self.x = self.x + list(xy[key][:-1])
            self.y = self.y + list(xy[key][1:])
        #self.x=xy[:-1] #t0 to t10 is the x file
        #self.y=xy[1:] #t1 to t11 is the y file
        self.x, self.y = torch.from_numpy(np.array(self.x)), torch.from_numpy(np.array(self.y)) # from list to array to torch tensor
        self.n_samples = self.x.shape[0] # number of samples = number of columns in xy

    def __getitem__(self, index):
        return self.x[index], self.y[index] # returns item at position 'index' and its target counter part

    def __len__(self):
        return self.n_samples
def create_dataset():
    dataset = Room_Sample_Dataset()
    return dataset

if __name__ == '__main__':
    dataset = Dataset_x4_y1()