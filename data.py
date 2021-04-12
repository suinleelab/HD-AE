from torch.utils.data import Dataset
from pandas import DataFrame, Series
import pdb

class DatasetWithConfounder(Dataset):
    def __init__(self, X: DataFrame, Z):
        self.X = X
        self.Z = Z

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if type(self.X) == DataFrame:
            X_ret = self.X.iloc[index].to_numpy()
        else:
            X_ret = self.X[index]

        if type(self.Z) == Series:
            Z_ret = self.Z.iloc[index]
        else:
            Z_ret = self.Z[index]
        return X_ret, Z_ret


class SimpleDataset(Dataset):
    def __init__(self, X: DataFrame):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index]
    
