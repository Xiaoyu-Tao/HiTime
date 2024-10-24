import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

class TS_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Dataloader for ECG5000 dataset

        Args:
          data (nd.array): The training or test data from the ECG5000 dataset, with the first column being the
          label column and the remaining 140 columns being the time series.
        """
        x = data[:, 1:]
        y = data[:, 0] - 1    # -1 as classes should start with zero instead of 1
        Y = LabelEncoder().fit_transform(y)
        # reshape from (n_samples, seq_len) to (n_samples, 1, seq_len)
        self.X = torch.tensor(x.reshape((x.shape[0], 1, x.shape[1]))).float()
        # one-hot encode labels
        self.Y = nn.functional.one_hot(torch.tensor(Y).to(torch.long)).float()

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.Y[idx, :]


if __name__ == "__main__":
    dataset = 'ECG200'
    train = np.loadtxt('./data/' + dataset + '/' + dataset + '_TRAIN.txt')
    print("Train:", train.shape)

    training_data = TS_dataset(train)

    train_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)

    for batch in train_loader:
        x, y = batch
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        break
