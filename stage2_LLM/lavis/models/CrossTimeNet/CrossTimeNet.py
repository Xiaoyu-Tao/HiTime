from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lavis.models.CrossTimeNet.VQVAE import VQVAE
from lavis.models.CrossTimeNet.FeatProcessor import FeatProcessor

from transformers import (
    BertForMaskedLM,
    BertConfig,
    BertTokenizer
)
import random
import os

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
    
class Model(nn.Module):
    def __init__(self, configs, task_type):
        super(Model, self).__init__()
        self.device = 'cuda:0'
        self.configs = configs
        self.feat_processor = FeatProcessor()
        
        self.data_name = configs.pretrain_dataset_list #args.data_path
        
        self.specific_data_id = configs.pretrain_dataset_list.index(configs.model_id)
        
        self.datasetid = {}
        self.feat_dim, self.seq_len, self.num_classes = [], [], []
        self.wave_length, self.n_embd_num, self.patch_num = [], [], []
        
        self.num_features = 768
        
        for idx, dataset in enumerate(configs.pretrain_dataset_list):
            self.datasetid.update({dataset:idx})
            s_len, f_dim, n_classes, w_length = self.feat_processor.get_data_attribute(dataset, wave_rate=0.1)
            
            self.seq_len.append(s_len)
            self.feat_dim.append(f_dim)
            self.num_classes.append(n_classes)
            self.wave_length.append(w_length)
            
            self.n_embd_num.append(1024)
            self.patch_num.append(s_len // w_length)
        
        self.emb_layer = nn.ModuleDict({})
        self.vqvae_model = nn.ModuleDict({})
        self.head = nn.ModuleDict({})
        
        
        config = BertConfig.from_pretrained(configs['local_model_path'], output_hidden_states=True)
        self.d_model = config.hidden_size
        self.dropout = 0.1
        
        self.patch_num = [self.seq_len[i] // self.wave_length[i] for i in range(len(self.seq_len))]#self.seq_len // self.wave_length
        self.custom_embeddings_dim = [64]*len(configs.pretrain_dataset_list)
        
        for name,seq_len,feat_dim,custom_embeddings_dim,n_embed,wave_length in zip(self.data_name,self.seq_len,self.feat_dim,self.custom_embeddings_dim,self.n_embd_num,self.wave_length):
            self.vqvae_model[name] = VQVAE(data_shape=(seq_len, feat_dim), hidden_dim=custom_embeddings_dim, n_embed=n_embed , wave_length=wave_length,block_num=4)

        self.n_embed = sum(self.n_embd_num)
        local_model_path = configs["local_model_path"] #"./bert"
        
        self.mask = nn.Parameter(torch.zeros(config.hidden_size))
        self.n_embed += 1
        self.mask_token = self.n_embed - 1
        nn.init.uniform_(self.mask, -1.0 / self.n_embed, 1.0 / self.n_embed)
        
        # if self.parms: #如果有参数
        self.encoder = BertForMaskedLM.from_pretrained(local_model_path,output_attentions=True, output_hidden_states=True)
        # else:
        #     self.encoder = BertForMaskedLM(config) # bert large
            
        weight = self.encoder.get_input_embeddings().weight
        sample = random.choices(list(range(len(weight))),k=self.n_embed)
        weight = weight[sample]
        self.emb_layer = nn.Embedding(self.n_embed, self.d_model).from_pretrained(weight)
        
        # Last Layer init
        self.encoder.config.vocab_size = self.n_embed
        new_output = nn.Linear(config.hidden_size, self.n_embed, bias=False)
        self.encoder.set_output_embeddings(new_output)

        for idx,i in enumerate(self.data_name):
            self.head[i] = nn.Sequential(*[nn.GELU(),nn.LayerNorm(self.d_model * self.patch_num[idx]),nn.Linear(self.d_model * self.patch_num[idx], self.num_classes[idx])])
    
    @staticmethod
    def init_weights_kaiming(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def init_vqvae(self, vqvae_model_path, dataset_name):
        vqvae_state_dict = torch.load(vqvae_model_path, map_location=self.device)

        self.vqvae_model[dataset_name].load_state_dict(vqvae_state_dict)
        self.vqvae_model[dataset_name].eval()
         
    def forward_feature(self, x_enc):
        B, L, M = x_enc.shape
        
        dataid = self.specific_data_id

        with torch.no_grad():
            _, _, labels = self.vqvae_model[self.data_name[dataid]](x_enc)
            if dataid > 0:
                offset = sum(self.n_embd_num[:dataid])
            else:
                offset = 0
            labels = labels + offset

        outputs = self.emb_layer(labels)
        
        self.encoder.eval()
        with torch.no_grad():
            outputs = self.encoder(inputs_embeds=outputs).hidden_states[-1]
            
        # print('outputs: ', outputs.shape)

        return outputs
    
def create_CrossTimeNet(configs, precision="fp16"):
    embeding = "word_mapping"
    model = Model(configs, embeding)

    print("Begin to load pretrained Encoder...")
    
    for dataset in configs.pretrain_dataset_list:
        vqvae_model_path = f"{configs['encoder_pretrained_folder']}//CrossTimeNet//pretrained_vae//{dataset}//model.pkl"
        
        # print(vqvae_model_path)
        model.init_vqvae(vqvae_model_path, dataset)
    
    model_path = os.path.join(configs['encoder_pretrained_folder'], 'CrossTimeNet', f"CrossTimeNet_{configs.pretrained_data}.pth")
    
    # load pretrain
    state_dict = torch.load(model_path, map_location="cpu")    
    model.load_state_dict(state_dict['model_state_dict'])
    
    print("Successfully load pretrained CrossTimeNet...")
    
    return model