import torch
import torch.nn as nn
import numpy as np


class MovingAvg(nn.Module):
    def __init__(self, input_shape: tuple, window_size: int):
        """
        Takes a batch of sequences with [batch size, channels, seq_len] and smoothes the sequences.
        Output size of moving average is: time series length - window size + 1

        Args:
            input_shape (tuple): input shape for the transformation layer in format (n_channels, length_of_timeseries)
            window_size (int): window size with which the time series is smoothed
        """
        assert len(input_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"
        super(MovingAvg, self).__init__()

        self.num_dim, self.length_x = input_shape
        self.window_size = window_size

        # compute output shape after smoothing (len ts - window size + 1)
        new_length = self.length_x - self.window_size + 1
        self.output_shape = (self.num_dim, new_length)

        # kernel weights for average convolution
        self.kernel_weights = torch.ones((self.num_dim, 1, self.window_size), dtype=torch.float) / self.window_size

    def forward(self, x):
        """
        Args:
          x (tensor): batch of time series samples
        Returns:
          output (tensor): smoothed time series batch
        """
        output = nn.functional.conv1d(x, self.kernel_weights, groups=self.num_dim)

        return output


class Downsample(nn.Module):
    def __init__(self, input_shape: tuple, sample_rate: int):
        """
        Takes a batch of sequences with [batch size, channels, seq_len] and down-samples with sample
        rate k. Hence, every k-th element of the original time series is kept.

        Args:
            input_shape (tuple): input shape for the transformation layer in format (n_channels, length_of_timeseries)
            sample_rate (int): sample rate with which the time series should be down-sampled
        """
        assert len(input_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"
        super(Downsample, self).__init__()

        self.sample_rate = sample_rate

        # compute output shape after down-sampling
        self.num_dim, self.length_x = input_shape
        last_one = 0
        if self.length_x % self.sample_rate > 0:
            last_one = 1
        new_length = int(np.floor(self.length_x / self.sample_rate)) + last_one
        self.output_shape = (self.num_dim, new_length)

    def forward(self, x):
        """
        Args:
          x (tensor): batch of time series samples
        Returns:
          output (tensor): down-sampled time series batch
        """
        batch_size = x.shape[0]

        last_one = 0
        if self.length_x % self.sample_rate > 0:
            last_one = 1

        new_length = int(np.floor(self.length_x / self.sample_rate)) + last_one
        output = torch.zeros((batch_size, self.num_dim, new_length))
        output[:, :, range(new_length)] = x[:, :, [i * self.sample_rate for i in range(new_length)]]

        return output


class Identity(nn.Module):
    def __init__(self, input_shape: tuple):
        """
        Identity mapping without any transformation (wrapper class).

        Args:
            input_shape (tuple): input shape for the transformation layer in format (n_channels, seq_len)
        """
        super(Identity, self).__init__()
        assert len(input_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"
        self.output_shape = input_shape

    def forward(self, x):
        """
        Args:
          x (tensor): batch of time series samples
        Returns:
          output (tensor): same as x
        """
        return x


if __name__ == "__main__":
    """ Some tests """

    # create data
    seq_len = 16
    data = torch.rand((32, 1, seq_len))
    print("Data shape:", data.shape)

    print("--- Test Down-sampling")
    print("Original length:", data.shape[2])
    sampling_rates = [2, 3, 4]
    for s in sampling_rates:
        ds_shape = Downsample(data.shape[1:], s).output_shape
        print("Rate {} -> Shape: {}".format(s, ds_shape))

    print("--- Test Smoothing")
    print("Original length:", data.shape[2])
    window_sizes = [3, 4, 5]
    for w in window_sizes:
        ma_shape = MovingAvg(data.shape[1:], w).output_shape
        print("Window size {} -> Shape: {}".format(w, ma_shape))
