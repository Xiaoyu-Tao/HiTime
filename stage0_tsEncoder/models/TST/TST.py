### Copy from tsai

import math, torch
import torch.nn as nn
import torch.nn.functional as F

from models.TST.layers import *
from models.TST.typing import *

def pv(text, verbose):
    if verbose: print(text)
    
def ifnone(a, b):
    "`b` if `a` is None else `a`"
    return b if a is None else a

class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k:int): 
        super(_ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:Optional[torch.Tensor]=None):

        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)                                         # scores : [bs x n_heads x q_len x q_len]
        
        # Scale
        scores = scores / (self.d_k ** 0.5)
        
        # Mask (optional)
        if mask is not None: scores.masked_fill_(mask, -1e9)
        
        # SoftMax
        attn = F.softmax(scores, dim=-1)                                    # attn   : [bs x n_heads x q_len x q_len]
        
        # MatMul (attn, v)
        context = torch.matmul(attn, v)                                     # context: [bs x n_heads x q_len x d_v]
        
        return context, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_k:int, d_v:int):
        r"""
        Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]
        """
        super(_MultiHeadAttention, self).__init__()
        
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        
        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:Optional[torch.Tensor]=None):
        
        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Scaled Dot-Product Attention (multiple heads)
        context, attn = _ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)          # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]

        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # context: [bs x q_len x n_heads * d_v]

        # Linear
        output = self.W_O(context)                                                  # context: [bs x q_len x d_model]
        
        return output, attn


def get_activation_fn(activation):
    if activation == "relu": return nn.ReLU()
    elif activation == "gelu": return nn.GELU()
    else: return activation()

class _TSTEncoderLayer(nn.Module):
    def __init__(self, q_len:int, d_model:int, n_heads:int, d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff:int=256, dropout:float=0.1, 
                 activation:str="gelu", enable_res_param=False, norm='batch'):
        super(_TSTEncoderLayer, self).__init__()
        assert d_model // n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)
        self.norm_tp = norm

        # Multi-Head attention
        self.self_attn = _MultiHeadAttention(d_model, n_heads, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), 
                                get_activation_fn(activation), 
                                nn.Dropout(dropout), 
                                nn.Linear(d_ff, d_model))
        
        self.enable = enable_res_param
        if self.enable:
            self.a = nn.Parameter(torch.tensor(1e-8))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

    def forward(self, src:torch.Tensor, mask:Optional[torch.Tensor]=None) -> torch.Tensor:

        # Multi-Head attention sublayer
        ## Multi-Head attention
        src2, attn = self.self_attn(src, src, src, mask=mask)
        ## Add & Norm
        
        src = src + self.dropout_attn(src2 * (self.a if self.enable else 1.)) # Add: residual connection with residual dropout
        
        # Norm: batchnorm or layernorm
        src = src.permute(0, 2, 1) if self.norm_tp == 'batch' else src
        src = self.norm_attn(src)      
        src = src.permute(0, 2, 1) if self.norm_tp == 'batch' else src
        
        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2 * (self.a if self.enable else 1.)) # Add: residual connection with residual dropout
        
        # Norm: batchnorm or layernorm
        src = src.permute(0, 2, 1) if self.norm_tp == 'batch' else src
        src = self.norm_ffn(src)      
        src = src.permute(0, 2, 1) if self.norm_tp == 'batch' else src

        return src


class _TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                 dropout=0.1, activation='gelu', n_layers=1, enable_res_param=False, norm='batch'):
        
        super(_TSTEncoder, self).__init__()
        self.layers = nn.ModuleList([_TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, 
                                                        activation=activation, enable_res_param=enable_res_param, norm=norm) \
                                                        for i in range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers: output = mod(output)
        return output

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=None, 
                 n_heads:int=16, d_k:Optional[int]=None, d_v:Optional[int]=None,  
                 act:str="gelu", pooling_tp='cat', fc_dropout:float=0., norm='batch',
                 y_range:Optional[tuple]=None, verbose:bool=False, device:str='cuda:0'):
        r"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:

        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        c_in, c_out, seq_len = configs.enc_in, configs.num_class, configs.seq_len
        n_layers, d_model, d_ff = configs.e_layers, configs.d_model, configs.d_ff
        dropout = configs.dropout
        
        super(Model, self).__init__()
        
        self.c_out, self.seq_len = c_out, seq_len
        
        # Input Embedding
        self.W_P = nn.Linear(c_in, d_model) # Eq 1: projection of feature vectors onto a d-dim vector space

        # Positional encoding
        W_pos = torch.empty((seq_len, d_model), device=device)
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = _TSTEncoder(seq_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=act, n_layers=n_layers, norm=norm)
        
        self.flatten = nn.Flatten()
        
        # Head
        self.head_nf = seq_len * d_model if pooling_tp == 'cat' else d_model
        self.head = self.create_head(self.head_nf, c_out, act=act, pooling_tp=pooling_tp, fc_dropout=fc_dropout, y_range=y_range)

    def create_head(self, nf, c_out, act="gelu", pooling_tp='cat', fc_dropout=0., y_range=None, **kwargs):
        layers = []
        if pooling_tp == 'cat':
            layers = [get_activation_fn(act), self.flatten]
            if fc_dropout: layers += [nn.Dropout(fc_dropout)]
        elif pooling_tp == 'mean':
            layers = [nn.AdaptiveAvgPool1d(1), self.flatten]
        elif pooling_tp == 'max':
            layers = [nn.AdaptiveMaxPool1d(1), self.flatten]
        
        layers += [nn.Linear(nf, c_out)]
        
        # could just be used in classifying task
        assert y_range == None
        return nn.Sequential(*layers)    
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) -> torch.Tensor:  # x: [bs x nvars x q_len]

        # Input encoding
        u = self.W_P(x_enc) # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        u = self.dropout(u + self.W_pos[:u.shape[1]]) 

        # Encoder
        z = self.encoder(u)                                             # z: [bs x q_len x d_model]
        z = z.transpose(2,1).contiguous()                               # z: [bs x d_model x q_len]

        # Classification/ Regression head
        return self.head(z)                                             # output: [bs x c_out]
