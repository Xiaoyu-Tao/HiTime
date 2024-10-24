import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, configs, activation=nn.GELU(), dropout=0.1):
        super(Model, self).__init__()
        self.name = 'MLP'
        hidden_sizes=[configs.d_model] * configs.e_layers
        input_size = configs.enc_in
        output_size = configs.num_class
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        
        for i in range(len(hidden_sizes)):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_sizes[i-1] if i > 0 else input_size, hidden_sizes[i]),
                nn.LayerNorm(hidden_sizes[i]),
                activation,
                nn.Dropout(dropout)
            ))
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.output_layer = nn.Linear(hidden_sizes[i-1], output_size)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x = x * 3
        # x = self.flatten(x)
        x = x_enc
        for k, layer in enumerate(self.layers):
            x = layer(x)
            
        x = self.pooling(x.permute(0,2,1)).reshape(x.shape[0], -1)
        x = self.output_layer(x) # B, Class 
        return x