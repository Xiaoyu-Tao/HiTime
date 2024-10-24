import os
import numpy as np
import pandas as pd
import torch


def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for classification or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """
    
    # print(batch_size)
    # print(data[0])
    
    features, labels, texts= zip(*data)
    
    # print(type(features))
    # print(type(labels))
    # print(type(texts))
    
    features = torch.stack(features, dim=0)
    labels = torch.cat(labels)
    texts = list(texts)
    
    return features, labels, texts


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', data_type='df', axis=None, \
                        mean=None, std=None, min_val=None, max_val=None, range=(-1,1)):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
            axis: optional (e.x. (0,1)), for calculate values of mean, std and so on.
            range: value range of minmax scaler
        """

        self.norm_type = norm_type
        self.data_type = data_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.range = range
        assert self.range in [(-1,1), (0,1)], "range of scaler should be in [(-1,1), (0,1)]"
        
        self.axis = axis

    def normalize(self, data):
        """
        Args:
            data: input dataframe or numpy
        Returns:
            data: normalized dataframe or numpy
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = data.mean() if self.data_type == 'df' else np.mean(data, axis=self.axis, keepdims=True)
                self.std = data.std() if self.data_type == 'df' else np.std(data, axis=self.axis, keepdims=True)
            return (data - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = data.max() if self.data_type == 'df' else np.max(data, axis=self.axis, keepdims=True)
                self.min_val = data.min() if self.data_type == 'df' else np.min(data, axis=self.axis, keepdims=True)
                
            data = (data - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)
            if self.range == (-1,1): data = data * 2 - 1.0
            return data

        elif self.norm_type == "per_sample_std":
            grouped = data.groupby(by=data.index)
            return (data - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = data.groupby(by=data.index)
            min_vals = grouped.transform('min')
            return (data - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y
