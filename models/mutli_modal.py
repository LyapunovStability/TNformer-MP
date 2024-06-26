import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class FusionModule(nn.Module):
    def __init__(self, ct_dim, ts_dim, hid_dim):
        super(FusionModule).__init__()
        self.linears = nn.ModuleList([nn.Linear(ct_dim, hid_dim),
                                      nn.Linear(ts_dim, hid_dim),
                                      nn.Linear(hid_dim, hid_dim)])

        self.act = nn.ELU()
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, ct_emb, ts_emb):
        aux_emb = self.linears[0](ct_emb) + self.linears[1](ts_emb)
        aux_emb = self.act(aux_emb)
        out = self.linears[2](aux_emb) + ts_emb
        out = self.norm(out)
        
        return out



class TNformer_MP(nn.Module):
    def __init__(self, ct_dim, ts_dim, hid_dim, h=4, layer=2):
        super(TNformer_MP).__init__()
        self.h = h
        self.ct_dim = ct_dim
        self.ts_dim = ts_dim
        self.hid_dim = hid_dim
        
        
        self.tn_tokenizer = TN_Tokenizer(ct_dim, ts_dim, hid_dim, h)
        self.prompt_tokenizer = Prompt_Tokenizer(ct_dim, ts_dim, hid_dim, h)
        
        self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hid_dim, nhead=4, dim_feedforward=hid_dim * 2, dropout=0.1, activation='gelu'), 
                num_layers=layer
                )
        self.pred_layer = nn.Linear(hid_dim, 1)
    
    
    def forward(self, data, ct_emb, ts_emb):
        if ct_emb is None:
            multi_token = self.tn_tokenizer(data, ct_emb, ts_emb)
        else:
            multi_token = self.prompt_tokenizer(data, ts_emb)

        out = self.encoder(multi_token)
        pred = self.pred_layer(out)
        
        return pred

        



class TN_Tokenizer(nn.Module):
    def __init__(self, ct_dim, ts_dim, hid_dim, h=4):
        super(TN_Tokenizer, self).__init__()
        self.h = 4
        self.ct_dim = ct_dim
        self.ts_dim = ts_dim
        self.hid_dim = hid_dim
        
        self.ct_embedding = nn.Sequential(nn.Linear(ct_dim, hid_dim))

        self.kernel_param = nn.Parameter(torch.randn(hid_dim//self.h))
        self.fusion_module = FusionModule(ct_dim, ts_dim, hid_dim)
        

    def forward(self, data, ct_emb, ts_emb):
        ct_idx = data["ct_t"]
        ct_t = data["ct_t"]
        ts_t = data["ts_t"]
        b, _, _  = ts_emb.shape
        
        temp_ct_emb = torch.zeros(b, ts_emb.shape[1], self.hidden_dim).to(ts_emb.device)
        for i in range(len(ct_idx)):
            idx = ct_idx[i]
            ct_t_i = ct_t[i] # m
            ts_t_i = ts_t[idx] # n
            ct_emb_i = ct_emb[i] #
            ts_emb_i = ts_emb[idx]
            out_i = self.temp_kernel_fusion(ct_t_i, ts_t_i, ct_emb_i, ts_emb_i)
            temp_ct_emb[idx, :, :] += out_i
     
        out = self.fusion_module(temp_ct_emb, ts_emb)
        out = torch.mean(out, dim=1)
        return out

        
        
    def feature_padding(self, emb, max_l):
        b, l, d = emb.shape
        out = torch.zeros(b, max_l, d).to(emb.shape)
        out[:, :l, :d]
    
    def cmp_norm(self, ct_t, ts_t):
        n = ts_t.shape
        m = ct_t.shape
        ct_t = ct_t.unsqueeze(0).expand(-1, n)
        ts_t = ts_t.unsqueeze(0).expand(m, -1)
        norm = (ct_t - ts_t) * (ct_t - ts_t) # m, n
        norm = norm.unsqueeze(-1) # m, n, 1 
        
        return norm      
        
    def temp_kernel_fusion(self, ct_t, ts_t, ct_emb, ts_emb):
        n = ts_t.shape
        m = ct_t.shape
        m, d = ct_emb.shape
        ct_emb = self.ct_embedding(ct_emb)
        ct_emb = ct_emb.reshape(m, self.h, d//self.h)
        
        norm = self.cmp_norm(ct_t, ts_t)
        alpha = torch.log(1 + torch.exp(self.kernel_param))
        ct_emb_out = []
        
        for i in range(self.h):
            ct_emb_1 = self.scale_kernel(self, ct_emb, ts_emb, norm, alpha, scale=i)
            ct_emb_out.append(ct_emb_1)
        
        ct_emb_out = torch.cat(ct_emb_out, dim=-1) # n, d
   
        return ct_emb_out
    
      
    def scale_kernel(self, ct_emb, ts_emb, norm, alpha, scale=1):
        m, d = ct_emb.shape
        n, _ = ts_emb.shape
        alpha = alpha.unsqueeze(0).unsqueeze(0).expand(m, n, -1) # m, n, d
        w = torch.logsumexp(- scale * alpha * norm, dim=2)
        sim = torch.exp(-scale * alpha * norm - w)
        ct_emb_1 = torch.einsum("mnd,md->nd", sim, ct_emb.squeeze(0))
        
        return ct_emb_1
    
    

class Prompt_Tokenizer(nn.Module):
    def __init__(self, ct_dim, ts_dim, hid_dim, h=4):
        super(Prompt_Tokenizer, self).__init__()
        self.h = 4
        self.ct_dim = ct_dim
        self.ts_dim = ts_dim
        self.hid_dim = hid_dim
        
        self.linears = nn.ModuleList([nn.Linear(ct_dim, hid_dim),
                                      nn.Linear(ts_dim, hid_dim),
                                      nn.Linear(ct_dim, hid_dim)])
        self.fusion_module = FusionModule(ct_dim, ts_dim, hid_dim)
        
        self.ct_prompt = nn.Embedding(self.h, ct_dim)


    def forward(self, data, ts_emb):
        b, l, d = ts_emb.shape
        idx = torch.arange(1)
        temp_prompt_emb = torch.zeros(b, l, self.hid_dim).to(ts_emb.device)
        for i in range(self.h):
            ct_prompt = self.ct_prompt[i]
            alpha = torch.sigmoid(self.linears[0](ct_prompt) + self.linears[1](ts_emb))
            ct_prompt = self.linears[2](ct_prompt) * alpha
            temp_prompt_emb += ct_prompt
        
        out = self.fusion_module(temp_prompt_emb, ts_emb)
        return out





class TempFusion(nn.Module):
    def __init__(self, ct_dim, ts_dim, hid_dim):
        super(TempFusion).__init__()
        self.h = 4
        self.ct_dim = ct_dim
        self.ts_dim = ts_dim
        self.hid_dim = hid_dim
        
        self.ct_embedding = nn.Sequential(nn.Linear(ct_dim, hid_dim))

        self.kernel_param = nn.Parameter(torch.randn(hid_dim//self.h))
        self.fusion_module = FusionModule(hid_dim, ts_dim, hid_dim)
        
        self.ct_prompt = nn.Embedding(1, ct_dim)
        self.promp_param = nn.Parameter(torch.randn(hid_dim))
    
    def forward(self, data, ct_emb, ts_emb):
        ts_len = data["ts_len"]
        ct_num = data["ct_num"]
        ct_t = data["ct_t"]
        ts_t = data["ts_t"]
        b, _, _  = ts_emb.shape
        
        temp_ct_emb = torch.zeros(b, int(max(ts_len)), self.hidden_dim).to(ts_emb.device)
        for i in range(len(ts_len)):
            n = ts_len[i]
            m = ct_num[i]
            ct_t_i = ct_t[i][:n] # m
            ts_t_i = ts_t[i][:m] # n
            ct_emb_i = ct_emb[i] #
            ts_emb_i = ts_emb[i]
            out_i = self.temp_kernel_fusion(ct_t_i, ts_t_i, ct_emb_i, ts_emb_i)
            temp_ct_emb[i, :n, :] = out_i
     
        out = self.fusion_module(temp_ct_emb, ts_emb)

        return out
        
        
    def feature_padding(self, emb, max_l):
        b, l, d = emb.shape
        out = torch.zeros(b, max_l, d).to(emb.shape)
        out[:, :l, :d]
    
    def cmp_norm(self, ct_t, ts_t):
        n = ts_t.shape
        m = ct_t.shape
        ct_t = ct_t.unsqueeze(0).expand(-1, n)
        ts_t = ts_t.unsqueeze(0).expand(m, -1)
        norm = (ct_t - ts_t) * (ct_t - ts_t) # m, n
        norm = norm.unsqueeze(-1) # m, n, 1 
        
        return norm      
        
    def temp_kernel_fusion(self, ct_t, ts_t, ct_emb, ts_emb):
        n = ts_t.shape
        m = ct_t.shape
        m, d = ct_emb.shape
        ct_emb = self.ct_embedding(ct_emb)
        ct_emb = ct_emb.reshape(m, self.h, d//self.h)
        
        norm = self.cmp_norm(ct_t, ts_t)
        alpha = torch.log(1 + torch.exp(self.kernel_param))
        ct_emb_out = []
        
        for i in range(self.h):
            ct_emb_1 = self.scale_kernel(self, ct_emb, ts_emb, norm, alpha, scale=i)
            ct_emb_out.append(ct_emb_1)
        
        ct_emb_out = torch.cat(ct_emb_out, dim=-1) # n, d
        out = self.fusion_module(ct_emb_out, ts_emb)
        
        return ct_emb_out
    
      
    def scale_kernel(self, ct_emb, ts_emb, norm, alpha, scale=1):
        m, d = ct_emb.shape
        n, _ = ts_emb.shape
        alpha = alpha.unsqueeze(0).unsqueeze(0).expand(m, n, -1) # m, n, d
        w = torch.logsumexp(- scale * alpha * norm, dim=2)
        sim = torch.exp(-scale * alpha * norm - w)
        ct_emb_1 = torch.einsum("mnd,md->nd", sim, ct_emb.squeeze(0))
        
        return ct_emb_1
    
    
    def prompt_fusion(self, ts_emb):
        b, l, d = ts_emb.shape
        idx = torch.arange(1)
        ct_prompt = self.ct_prompt[idx]
        ct_prompt = self.ct_embedding(ct_prompt)

        alpha = torch.log(1 + torch.exp(self.kernel_param))
        ct_prompt = ct_prompt * alpha
        ct_prompt = ct_prompt.unsqueeze(0).expand(b, -1)
        out = self.fusion_module(ct_prompt, ts_emb)
        return out