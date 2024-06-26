import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from torch import nn
import torchvision

class CT_Encoder(nn.Module):
    def __init__(self, hid_dim=128):
        super(CT_Encoder, self).__init__()
        self.convnext = self.create_convnext_tiny()        
        self.linear =  nn.Linear(768, hid_dim)

    
    def create_convnext_tiny(self, num_classes=1):
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        model = torchvision.models.convnext_tiny(weights=weights)
        return model
  
    def forward(self, data):
        x = data["ct"]
        x = self.convnext.features(x)
        x = self.convnext.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class TS_Encoder(nn.Module):
    def __init__(self, hid_dim):
        super(TS_Encoder, self).__init__()
        self.convnext = self.create_convnext_tiny()        
        self.linear =  nn.Linear(hid_dim, hid_dim)

    
    def create_convnext_tiny(self, num_classes=1):
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        model = torchvision.models.convnext_tiny(weights=weights)
        return model
  
    def forward(self, data):
        x = data["ts"]
        x = self.convnext.features(x)
        x = self.convnext.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x    



class TS_Encoder(nn.Module):
    def __init__(self, hid_dim, input_size=46,  device="cuda:0"):
        super(TS_Encoder, self).__init__()

        self.device = device
        self.rnn_hid_size = hid_dim
        self.input_size = input_size

        self.build()

    def build(self):

        self.t_gru_x = nn.GRUCell(self.input_size, self.rnn_hid_size)
        self.t_gru_alpha = nn.GRUCell(self.input_size, self.rnn_hid_size)
        self.standard_gru = nn.GRUCell(self.input_size, self.rnn_hid_size)

        self.hist_reg_x = nn.Linear(self.rnn_hid_size, self.input_size)
        self.hist_reg_alpha = nn.Linear(self.rnn_hid_size, self.input_size)

        self.hist_reg = nn.Sequential(nn.Linear(self.input_size * 2, self.input_size), nn.ReLU())

        self.std_emb = nn.Sequential(nn.Linear(self.input_size, self.input_size), nn.Sigmoid())


    def forward(self, data):
        
        x = data["ts"]
        std = data["mask"]
        mask = data["mask"]
        time_stamp = data["ts_t"] 
        B, L, K = x.shape
        h_mask = (time_stamp).bool().float() # B, L
        h_mask = h_mask.unsqueeze(-1).expand(-1, -1, self.rnn_hid_size) # B, L, D

        h = torch.zeros((B, self.rnn_hid_size)).to(x.device)
        h_x = torch.zeros((B, self.rnn_hid_size)).to(x.device)
        h_alpha = torch.zeros((B, self.rnn_hid_size)).to(x.device)

        h_state = []
        B, L, K = x.shape
        for t in range(L):
            x_t = x[:, t, :]
            h_m = h_mask[:, t, :]
            c_t = 1 - std[:, t, :] #B,K
            if t == 0:
                d = torch.zeros_like(time_stamp[:, t])
            else:
                d = torch.abs(time_stamp[:, t] - time_stamp[:, t-1])
            decay_factor = 1/ torch.log(math.e + d.unsqueeze(-1))
            alpha_t = self.std_emb(c_t)
            x_t_u = x_t * alpha_t
            x_t_s = x_t * torch.relu(c_t)
            h_x = self.t_gru_x(x_t_s, h_x * decay_factor)
            h_alpha = self.t_gru_alpha(alpha_t, h_alpha * decay_factor)

            x_t_s_d = self.hist_reg_x(h_x)
            alpha_t_d = torch.sigmoid(self.hist_reg_alpha(h_alpha))

            x_t_s_1 = x_t_s_d * alpha_t_d

            x_t_adjust = self.hist_reg(torch.cat([x_t_s_1, x_t_u], dim=-1))
            h = self.standard_gru(x_t_adjust, h)
            h_state.append(h)
            h = h * h_m + (1 - h_m) * h_state[-1]

        h = torch.stack(h_state, dim=1)

        return h

class Uni_Pred_CT(nn.Module):
    def __init__(self, hid_dim=128):
        super(Uni_Pred_CT, self).__init__()    
        self.linear =  nn.Linear(1, hid_dim)
        self.pred_layer =  nn.Linear(hid_dim, 1)

    def temporal_decay(self, d, h):

        gamma_h = self.linear(d)
        gamma_h = F.relu(gamma_h)
        gamma_h = torch.exp(-gamma_h)
        h = h * gamma_h
        return h
    
    
    def forward(self, data, ct):
        delta_t = data["ct_t"]
        ct = self.temporal_decay(delta_t, ct)
        y = self.linear(ct)
        return y 

