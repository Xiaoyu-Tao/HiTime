import os
import pickle
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')
    
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

from data_provider.gen_text import TextGenerator

warnings.filterwarnings('ignore')

class UEAloader(Dataset):
    def __init__(self, dataset, root_path, file_list=None, limit_size=None, flag=None, **kwargs):
        self.kwargs = kwargs
        self.dataset = dataset
        self.root_path = root_path
        
        self.feature_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        
        # use all features
        self.dims
        self.class_names
        
        self.text_generator = TextGenerator(dataset)
        # self.min_val, self.max_val
        
        # pre_process
        # normalizer = Normalizer()
        # self.feature_df = normalizer.normalize(self.feature_df)
        print(self.length)
        
    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        data_p, label_p = None, None
        if flag == 'TRAIN':
            data_p = os.path.join(root_path, 'train_d.npy')
            label_p = os.path.join(root_path, 'train_l.npy')
        elif flag in ['VAL', 'TEST']:
            data_p = os.path.join(root_path, 'test_d.npy')
            label_p = os.path.join(root_path, 'test_l.npy')
        else:
            raise Exception("No flag: {}, should be in 'TRAIN', 'VAL' or 'TEST'".format(flag))
        
        datas, labels = np.load(data_p), np.load(label_p)
        
        # normalizer = Normalizer(norm_type='minmax', data_type='numpy', axis=(0,1), \
        #             min_val=self.kwargs['min_val'], max_val=self.kwargs['max_val'])
        
        # datas = normalizer.normalize(datas)
        # self.min_val, self.max_val = normalizer.min_val, normalizer.max_val 
        
        labels = np.expand_dims(labels, axis=1)
        
        self.number_of_dataset = datas.shape[0]
        
        self.dims = datas.shape[-1]
        self.length = datas.shape[1]
        self.max_seq_len = datas.shape[1]
        self.class_names = np.unique(labels)
        
        # train data: (8823, 128, 9)
        # test data: (8823, ) 
        # test label: [0,1,2,3,4,5]

        return datas, labels 
    
    def __getitem__(self, ind):
        
        data, label = torch.from_numpy(self.feature_df[ind]), torch.Tensor(self.labels_df[ind, 0].flatten())
        prompt, target = self.text_generator.get_prompt_and_target(label)
        
        return data, label, target

    def __len__(self):
        return self.number_of_dataset