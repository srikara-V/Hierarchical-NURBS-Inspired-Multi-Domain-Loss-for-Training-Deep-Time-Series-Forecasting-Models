import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

    


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.d_ff
        self.channels = configs.enc_in
        self.individual = configs.channel_independence
        self.num_layers = 3

        self.gelu = nn.GELU()
        
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(self.seq_len, self.hidden_dim))
        for _ in range(self.num_layers - 2):
            self.linear_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.linear_layers.append(nn.Linear(self.hidden_dim, self.pred_len))

    def forward(self, x, *args, **kwargs):
            for linear in self.linear_layers[:-1]:
                x = linear(x.permute(0, 2, 1)).permute(0, 2, 1)
                x = self.gelu(x)
            x = self.linear_layers[-1](x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
 