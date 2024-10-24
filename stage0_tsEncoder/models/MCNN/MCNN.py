import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
	def __init__(self, configs, device = 'cuda:0'):
		"""
		Multi-Scale Convolutional Neural Network for Time Series Classification - Cui et al. (2016).

		Args:
		  ts_shape (tuple):           shape of the time series, e.g. (1, 9) for uni-variate time series
									  with length 9, or (3, 9) for multivariate time series with length 9
									  and three features
		  n_classes (int):            number of classes
		  pool_factor (int):          length of feature map after max pooling, usually in {2,3,5}
		  kernel_size (int or float): filter size for convolutional layers, usually ratio in {0.05, 0.1, 0.2}
									  times the length of time series
		  transformations (dict):     dictionary with key value pairs specifying the transformations
									  in the format 'name': {'class': <TransformClass>, 'params': <parameters>}
		"""
		super(Model, self).__init__()
		ts_shape, n_classes = (configs.enc_in, configs.seq_len), configs.num_class

		assert len(ts_shape) == 2, "Expecting shape in format (n_channels, seq_len)!"

		ts_length = ts_shape[-1]
		pool_factor = 4
		if ts_length in [30, 45]:
			pool_factor = 3
		elif ts_length in [22, 15, 29]:
			pool_factor = 2
		elif ts_length in [8]:
			pool_factor = 1

		transformations = {
			'identity': {
				'class': Identity,
				'params': []
			},
			'movingAvg': {
				'class': MovingAvg,
				'params': [3, 4, 5]       # window sizes
			},
			'downsample': {
				'class': Downsample,
				'params': [2, 3]       # sampling rates
			}
		}

		kernel_size = max(1, int(ts_length*0.05))

		
		self.name = 'MCNN'
		self.ts_shape = ts_shape
		self.n_classes = n_classes
		self.pool_factor = pool_factor
		self.kernel_size = int(self.ts_shape[1] * kernel_size) if kernel_size < 1 else int(kernel_size)

		self.loss = nn.CrossEntropyLoss

		# layer settings
		self.local_conv_filters = 64
		self.local_conv_activation = nn.ReLU  # nn.Sigmoid in original implementation
		self.full_conv_filters = 256
		self.full_conv_activation = nn.ReLU  # nn.Sigmoid in original implementation
		self.fc_units = 64
		self.fc_activation = nn.ReLU  # nn.Sigmoid in original implementation

		# setup branches
		self.branches = self._setup_branches(transformations)
		self.n_branches = len(self.branches)

		# full convolution
		in_channels = self.local_conv_filters * self.n_branches
		# kernel shouldn't exceed the length (length is always pool factor?)
		full_conv_kernel_size = int(min(self.kernel_size, int(self.pool_factor)))
		self.full_conv = nn.Conv1d(in_channels, self.full_conv_filters,
								   kernel_size=full_conv_kernel_size,
								   padding='same')
		# ISSUE: In the TensorFlow implementation of https://github.com/hfawaz/dl-4-tsc they implement the pool_size
		# as follows: pool_size = int(int(full_conv.shape[1])/pool_factor).
		# However, this makes no sense here, as the output length of the timeseries after convolution (denoted as n) is
		# equal to pool_factor due to the local max pooling operation. Dividing n/pool_factor always yields a
		# max pooling size of 1, which is simply identity mapping. E.g. the output shape after local convolution is
		# (batch_size, len_ts, n_channels) and the pooling factor ensures that len_ts is equal to pool_factor. Hence,
		# the shape after local pooling is (batch_size, pool_factor, n_channels) and shape[1]/pool_factor = 1.
		pool_size = 1
		self.full_conv_pool = nn.MaxPool1d(pool_size)

		# fully-connected
		self.flatten = nn.Flatten()
		in_features = int(self.pool_factor * self.full_conv_filters)
		self.fc = nn.Linear(in_features, self.fc_units, bias=True)

		# output
		self.output = nn.Linear(self.fc_units, self.n_classes, bias=False)

	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
		x_enc = x_enc.permute(0, 2, 1)

		xs = [self.branches[idx](x_enc) for idx in range(self.n_branches)]

		x = torch.cat(xs, dim=1)

		x = self.full_conv(x)
		x = self.full_conv_activation()(x)
		x = self.full_conv_pool(x)

		x = self.flatten(x)
		x = self.fc(x)
		x = self.fc_activation()(x)

		x = self.output(x)
		# x = self.softmax(x)

		return x

	def _build_local_branch(self, name: str, transform: nn.Module, params: list):
		"""
		Build transformation and local convolution branch.

		Args:
		  name (str):   Name of the branch.
		  transform (nn.Module):  Transformation class applied in this branch.
		  params (list):   Parameters for the transformation, with the first parameter always being the input shape.
		Returns:
		  branch:   Sequential model containing transform, local convolution, activation, and max pooling.
		"""
		branch = nn.Sequential()
		# transformation
		branch.add_module(name + '_transform', transform(*params))
		# local convolution
		branch.add_module(name + '_conv', nn.Conv1d(self.ts_shape[0], self.local_conv_filters,
													kernel_size=self.kernel_size, padding='same'))
		branch.add_module(name + '_activation', self.local_conv_activation())
		# local max pooling (ensure that outputs all have length equal to pool factor)
		pool_size = int(int(branch[0].output_shape[1]) / self.pool_factor)
		assert pool_size > 1, "ATTENTION: pool_size can not be 0 or 1, as the lengths are then not equal" \
							  "for concatenation!"
		branch.add_module(name + '_pool', nn.MaxPool1d(pool_size))  # default stride equal to pool size

		return branch

	def _setup_branches(self, transformations: dict):
		"""
		Setup all branches for the local convolution.

		Args:
		  transformations:  Dictionary containing the transformation classes and parameter settings.
		Returns:
		  branches: List of sequential models with local convolution per branch.
		"""
		branches = []
		for transform_name in transformations:
			transform_class = transformations[transform_name]['class']
			parameter_list = transformations[transform_name]['params']

			# create transform layer for each parameter configuration
			if parameter_list:
				for param in parameter_list:
					if np.isscalar(param):
						name = transform_name + '_' + str(param)
						branch = self._build_local_branch(name, transform_class, [self.ts_shape, param])
					else:
						branch = self._build_local_branch(transform_name, transform_class,
														  [self.ts_shape] + list(param))
					branches.append(branch)
			else:
				branch = self._build_local_branch(transform_name, transform_class, [self.ts_shape])
				branches.append(branch)

		return torch.nn.ModuleList(branches)

class MovingAvg(nn.Module):
	def __init__(self, input_shape: tuple, window_size: int, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
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
		self.kernel_weights = torch.ones((self.num_dim, 1, self.window_size), dtype=torch.float, device=device) / self.window_size

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
	def __init__(self, input_shape: tuple, sample_rate: int, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
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
		self.device = device

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
		output = torch.zeros((batch_size, self.num_dim, new_length), device=self.device)
		output[:, :, range(new_length)] = x[:, :, [i * self.sample_rate for i in range(new_length)]]

		return output


class Identity(nn.Module):
	def __init__(self, input_shape: tuple, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
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
	dsid = 'xxx'

	transformations = {
		'identity': {
			'class': Identity,
			'params': []
		},
		'movingAvg': {
			'class': MovingAvg,
			'params': [3, 4, 5]       # window sizes
		},
		'downsample': {
			'class': Downsample,
			'params': [2, 3]       # sampling rates
		}
	}

	classes = 9
	# seq_lens = [23, 15, 144, 640, 100, 182, 24, 1197, 270, 17984, 207, 65, 1751, 62, 50, 400, 152, 405, 
	# 30, 29, 45, 36, 3000, 51, 144, 8, 217, 30, 896, 1152, 93, 2500, 315]
	seq_lens = [22, 30]
	

	channels = 1

	for seq_len in seq_lens:
		kernel_size = int(seq_len*0.05)
		if kernel_size < 1:
			kernel_size = 1
		pool_factor = 4
		if seq_len in [30, 45]:
			pool_factor = 3
		elif seq_len in [22, 15, 29]:
			pool_factor = 2
		elif seq_len in [8]:
			pool_factor = 1
		
		print("Model with {} classes, sequence length of {}, and {} channels (multivariate)".format(
			classes, seq_len, channels
		))

		model = MCNN(0, 0, (channels, seq_len), classes, pool_factor, kernel_size, transformations).cuda()

		print("--- Model:\n", model)

		# forward pass
		data = torch.rand((64, channels, seq_len), device='cuda')
		print("--- Input shape:", data.shape)
		print("--- Model sample output:", model.forward(data).shape)
