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
class Dataset_x4_y1(Dataset):
    def __init__(self):
        #load npz file

        # if true => deviation of the data is calculated and used, if false the original data is used
        xy = import_all_rooms(False)
        self.x = []
        self.y = []

        for key, _ in xy.items(): # key = name of experiment
            samples_4 = []
            for i in range(56):
                samples_4.append([]) # for each set of 4 we need a new list
                for j in range(4):
                    samples_4[i] += list(xy[key][i + j])  # 4 temp matrices of 4 timesteps t,t-1,t-2,t-3
            self.x += samples_4

            self.y = self.y + list(xy[key][4:]) # i think 4

        self.x = np.expand_dims(self.x, axis=2) # add another dimension for the channels (n_samples, 4, 61, 81, 31) --> (n_samples, 4, 1, 61, 81, 31)
        # self.x dimensions = (n_samples, 4, 1, 61, 81, 31)
        # self.y dimensions = (n_samples, 1, 61, 81, 31)

        self.x, self.y = torch.from_numpy(np.array(self.x)), torch.from_numpy(np.array(self.y)) # from list to array to torch tensor
        self.n_samples = self.x.shape[0] # number of samples = number of columns in xy
        # print(r"xshape="+str(self.n_samples)+ '\n'+'yshape=' +str(self.y.shape[0]))

    def __getitem__(self, index):
        # dataset[idx] = (input = (4,1,x,y,z), target = (1,x,y,z) )
        return self.x[index], self.y[index] # returns item at position 'index' and its target counter part


    def __len__(self):
        return self.n_samples
def create_dataset():
    dataset = Room_Sample_Dataset()
    return dataset



if __name__ == '__main__':


    dataset = Dataset_x4_y1()




