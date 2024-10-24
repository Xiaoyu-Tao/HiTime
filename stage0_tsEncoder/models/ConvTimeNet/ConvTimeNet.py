import torch, math
import numpy as np
import torch.nn as nn

from models.ConvTimeNet.dlutils import DeformablePatch, get_activation_fn

from models.ConvTimeNet.ConvTimeNet_backbone import ConvTimeNet_backbone

class Model(nn.Module):
	def __init__(self, configs):
		super().__init__()
		self.name = 'Model'
		patch_size, patch_stride = configs.patch_size, int(configs.patch_size * 0.5)
  
		# DePatch Embedding
		in_channel, out_channel, seq_len = configs.enc_in, configs.d_model, configs.seq_len
		self.depatchEmbedding = DeformablePatch(in_channel, out_channel, seq_len, patch_size, patch_stride)
  
		# ConvTimeNet Backbone
		new_len = self.depatchEmbedding.new_len
		c_in, c_out, dropout = out_channel, configs.num_class, configs.dropout,
		d_ff, d_model, dw_ks = configs.d_ff, configs.d_model, configs.dw_ks
  
		block_num, enable_res_param, re_param = len(dw_ks), True, True
		
		self.main_net = ConvTimeNet_backbone(c_in, c_out, new_len, block_num, d_model, d_ff, 
						  dropout, act='gelu', dw_ks=dw_ks, enable_res_param=enable_res_param, 
						  re_param=re_param, norm='batch', use_embed=False, device='cuda:0')
  
		# linear proj
		# d_qformer = configs.d_qformer
		# self.linear_proj = nn.Linear(d_model, d_qformer)
		# self.proj_dropout = nn.Dropout(dropout)
  
		# Head
		d_qformer = d_model
		layers = [nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(d_qformer, c_out)]
		self.head = nn.Sequential(*layers)  

	def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
		out_patch = self.depatchEmbedding(x_enc) # [bs, ffn_output(before softmax)]
		z = self.main_net(out_patch.permute(0, 2, 1))	
		output = self.head(z)  
		# print(z.shape)
		# print(self.linear_proj)
  
		# proj_z = self.proj_dropout(self.linear_proj(z.permute(0, 2, 1)))
		# output = self.head(proj_z.permute(0, 2, 1))     
		return output