import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *shape): 
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], -1) if not self.shape else x.reshape(-1) if self.shape == (-1,) else x.reshape(x.shape[0], *self.shape)
    def __repr__(self): return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"

class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()
    def forward(self, x):
        return self.flatten(self.gap(x))

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(ni,nf,ks,stride=stride,padding=padding,dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(nf,nf,ks,stride=stride,padding=padding,dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, 
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(ni,nf,1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) # out

def TemporalConvNet(c_in, layers, ks=2, dropout=0.):
    temp_layers = []
    for i in range(len(layers)):
        dilation_size = 2 ** i
        ni = c_in if i == 0 else layers[i-1]
        nf = layers[i]
        temp_layers += [TemporalBlock(ni, nf, ks, stride=1, dilation=dilation_size, padding=(ks-1) * dilation_size, dropout=dropout)]
    return nn.Sequential(*temp_layers)


class Model(nn.Module):
    def __init__(self, configs, ks=5, conv_dropout=0.1, fc_dropout=0.1):
        super(Model, self).__init__()
        layers=[configs.d_model] * configs.e_layers
        
        c_in, c_out = configs.enc_in, configs.num_class

        self.name = 'TCN'
        self.tcn = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1],c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = x_enc.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)