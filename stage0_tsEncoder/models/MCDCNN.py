import torch
import torch.nn as nn
import numpy as np
import copy
from math import floor

class Model(nn.Module):
	def __init__(self, configs):
		super().__init__()
		ts_shape, n_classes = (configs.enc_in, configs.seq_len), configs.num_class

		self.name = 'MCDCNN'
		self.ts_shape = ts_shape # channels * seq_len
		self.n_classes = n_classes
		n_channels = ts_shape[0]
		seq_lens = ts_shape[1]
		assert len(ts_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"
		self.channels = nn.ModuleList(Channel(ts_shape) * n_channels)

		self.fc1 = nn.Linear(n_channels * floor(floor(seq_lens / 2)/ 2) * 64, 64)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(64, n_classes)
		# self.softmax = nn.Softmax(dim=1)

	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
		x_enc = x_enc.permute(0, 2, 1)
		outs = []
		for i, channel in enumerate(self.channels):
			out = channel(x_enc[:, i:i+1, :])
			outs.append(out)

		outs = torch.cat(outs, dim=-1)

		outs = self.relu(self.fc1(outs))
		output = self.fc2(outs)

		return output


class Channel(nn.Module):
	def __init__(self, input_shape) -> None:
		super().__init__()
		self.input_shape = input_shape
		self.layers = nn.Sequential(
			nn.Conv1d(1, 64, kernel_size=5, padding='same'),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Conv1d(64, 64, kernel_size=5, padding='same'),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Flatten()
		)

	def __mul__(self, other):
		"""
		Overrides operator '*' for Channel class. Now, it allows for the rapid creation
		of multiple channels.
		Parameters
		----------
		other : int
			Number of identical channels to be created, has to be > 0

		Returns
		-------
		new_channels : list
					List of 'other' number of deepcopies of original Channel
		"""
		if isinstance(other, int) and other > 0:
			return [copy.deepcopy(self) for _ in range(other)]
		else:
			raise TypeError(f"Value: {other} is not an integer. Channels can only be multiplied by integers")

	def forward(self, x_enc):
		return self.layers(x_enc)

if __name__ == '__main__':
	x = torch.rand((300, 2, 8))
	ts_length = x.shape[-1]
	ts_shape = (x.shape[-2], x.shape[-1])
	input_shape = (1, x.shape[-1])
	n_classes = 4

	# channel = Channel(input_shape)
	# output = channel(x)
	# print(output.shape)
	mcdcnn = Model(0, 0, ts_shape, n_classes)

	print(mcdcnn)
	print(mcdcnn(x).shape)

