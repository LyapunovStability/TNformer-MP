from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys; sys.path.append('..')
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime, timedelta
import time

import numpy as np
from sklearn import metrics

from dataset_uni import get_img_dataloader, get_ts_dataloader
from dataset_multi import get_multi_dataloader

from models.uni_modal import CT_Encoder, TS_Encoder, Uni_Pred_CT

from models.mutli_modal import TNformer_MP


class Trainer():
    def __init__(self, config):
        self.config = config
        
        self.ct_batch_size = self.config["train"]["ct_batch_size"]
        self.ts_batch_size = self.config["train"]["ts_batch_size"]
        self.ct_epoch = self.config["train"]["ct_epoch"]
        self.ts_epoch = self.config["train"]["ts_epoch"]
        self.multi_epoch = self.config["train"]["multi_epoch"]
        self.uni_lr_ct = self.config["train"]["uni_lr_ct"]
        self.uni_lr_ts = self.config["train"]["uni_lr_ts"]
        self.device = self.config["train"]["device"]
        
        
        self.ct_dim = self.config["model"]["ct_dim"]
        self.ts_dim = self.config["model"]["ts_dim"]
        self.hid_dim = self.config["model"]["hid_dim"]
        
        self.time_start = time.time()
        self.time_end = time.time()
        self.start_epoch = 1
        self.patience = 0
        self.ct_encoder = CT_Encoder(self.ct_dim).to(self.device)
        self.ts_encoder = TS_Encoder(self.ts_dim).to(self.device)
        self.ct_pred_layer = Uni_Pred_CT(self.ct_dim).to(self.device)
        self.ts_pred_layer = nn.Linear(self.ts_dim, 1).to(self.device)
        self.fusion_model = TNformer_MP(self.ct_dim, self.ts_dim, self.hid_dim).to(self.device)
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def get_ct_dataset(self):
        train_loader, val_loader, test_loader = get_img_dataloader(self.config["train"]["ct_batch_size"])
        return train_loader, val_loader, test_loader
    
    def get_ts_dataset(self):
        train_loader, val_loader, test_loader = get_ts_dataloader(self.config["train"]["ts_batch_size"])
        return train_loader, val_loader, test_loader
    
    
    def get_multimodal_dataset(self):
        train_loader_pair, val_loader_pair, test_loader_pair, train_loader_miss, val_loader_miss, test_loader_miss = get_multi_dataloader(self.config["train"]["multi_batch_size"], self.config["train"]["ts_batch_size"])
        return train_loader_pair, val_loader_pair, test_loader_pair, train_loader_miss, val_loader_miss, test_loader_miss
    
    def process_data(self, batch):
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)

        return batch
   
    
    def uni_modal_train_ct(self):
        
        train_loader, val_loader, test_loader = self.get_ct_dataset()
        
        optimizer = optim.SGD([self.ct_encoder.parameters(),self.ct_pred_layer.parameters()], lr=self.uni_lr_ct)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.ct_epoch, verbose=True)
        
        for epoch in range(self.ct_epoch):
            self.ct_encoder.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch = self.process_data(batch)
                y = batch["y"]
                ct_emb = self.ct_encoder(batch)
                pred = self.ct_pred_layer(batch, ct_emb)
                loss = self.bce_loss(pred, y)
                loss.backward()
                optimizer.step()
            lr_scheduler.step()  
            self.ct_encoder.eval()  
            self.uni_validate_ct(val_loader, epoch)
            
        print("CT training done")
        self.uni_validate_ct(test_loader, epoch)
        
        
    def uni_validate_ct(self, val_loader, epoch=-1):
        all_pred = []
        all_y = []
        for batch in val_loader:
            batch = self.process_data(batch)
            y = batch["y"]
            ct_emb = self.ct_encoder(batch)
            pred = self.ct_pred_layer(batch, ct_emb)
            pred = torch.sigmoid(pred)
            all_pred.append(pred)
            all_y.append(y)
        all_pred = torch.cat(all_pred, dim=0).cpu().detach().numpy()
        all_y = torch.cat(all_y, dim=0).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(all_y, all_pred)
        auprc = metrics.average_precision_score(all_y, all_pred)
        print(f"Epoch {epoch} AUROC: {auroc} AUPRC: {auprc}")
        
        return auroc, auprc
                
    
    def uni_modal_train_ts(self):
        train_loader, val_loader, test_loader = self.get_ts_dataset()
        
        optimizer = optim.Adam([self.ts_encoder.parameters(),self.ts_pred_layer.parameters()], lr=self.uni_lr_ts)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.ts_epoch, verbose=True)
        
        for epoch in range(self.ct_epoch):
            self.ct_encoder.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch = self.process_data(batch)
                y = batch["y"]
                ts_emb = self.ts_encoder(batch)
                ts_emb = torch.mean(ts_emb, dim=1)
                pred = self.ts_pred_layer(ts_emb)
                loss = self.bce_loss(pred, y)
                loss.backward()
                optimizer.step()
                
            lr_scheduler.step()
            self.ts_encoder.eval()  
            self.uni_validate_ts(val_loader, epoch)
    
        print("TS training done")
        self.uni_validate_ts(test_loader, epoch)
             
        
    def uni_validate_ts(self, val_loader, epoch=-1):
        all_pred = []
        all_y = []
        for batch in val_loader:
            batch = self.process_data(batch)   
            y = batch["y"]
            ts_emb = self.ts_encoder(batch)
            ts_emb = torch.mean(ts_emb, dim=1)
            pred = self.ts_pred_layer(ts_emb)
            pred = torch.sigmoid(pred)
            all_pred.append(pred)
            all_y.append(y)
        all_pred = torch.cat(all_pred, dim=0).cpu().detach().numpy()
        all_y = torch.cat(all_y, dim=0).cpu().detach().numpy()
        auroc = metrics.roc_auc_score(all_y, all_pred)
        auprc = metrics.average_precision_score(all_y, all_pred)
        print(f"Epoch {epoch} AUROC: {auroc} AUPRC: {auprc}")
        
        return auroc, auprc
    
    
    
    def multi_modal_train(self):
        
        train_loader_pair, val_loader_pair, test_loader_pair, train_loader_miss, val_loader_miss, test_loader_miss = self.get_multimodal_dataset()
        optimizer = optim.Adam([{"params": self.ts_encoder.parameters(), "lr": 0.0001},
                                {"params": self.ct_encoder.parameters(), "lr": 0.0001},
                                {"params": self.fusion_model.parameters(), "lr": 0.0001}])
        
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.multi_epoch, verbose=True)
        
        for epoch in range(self.multi_epoch):
            self.ct_encoder.train()
            self.ts_encoder.train()
            self.fusion_model.train()
            num = min(len(train_loader_pair),len(train_loader_miss))
            for _ in range(num):
                optimizer.zero_grad()
                # ---- multi-modal training ----
                batch_pair = next(train_loader_pair)
                batch_pair = self.process_data(batch_pair)
                y = batch_pair["y"]
                ct_emb = self.ct_encoder(batch_pair)
                ts_emb = self.ts_encoder(batch_pair)
                pred = self.fusion_model(batch_pair, ct_emb, ts_emb)
                loss_pair = self.bce_loss(pred, y)
                
                # ---- missing-modal training ----
                batch_miss = next(train_loader_miss)
                batch_miss = self.process_data(batch_miss)
                y = batch_miss["y"]
                ts_emb = self.ts_encoder(batch_miss)
                pred = self.fusion_model(batch_miss, None, ts_emb)
                loss_miss = self.bce_loss(pred, y)
                
                loss = loss_pair + loss_miss
                loss.backward()
                optimizer.step()
            
            lr_scheduler.step()
              
            self.ts_encoder.eval()
            self.ct_encoder.eval()
            self.fusion_model.eval()
            self.multi_validate(val_loader_pair, val_loader_miss, epoch)
    
        print("Multi-modal training done")
        self.multi_validate(test_loader_pair, test_loader_miss, epoch)
        
        
        
    def multi_validate(self, val_loader_pair, val_loader_miss, epoch=-1):
        all_pred = []
        all_y = []
        all_pred_pair = []
        all_y_pair = []
        all_pred_miss = []
        all_y_miss = []
        for batch in val_loader_pair:
            y = batch["y"]
            batch = self.process_data(batch)
            ts_emb = self.ts_encoder(batch)
            ct_emb = self.ct_encoder(batch)
            pred = self.fusion_model(batch, ct_emb, ts_emb)
            pred = torch.sigmoid(pred)
            all_pred_pair.append(pred)
            all_y_pair.append(y)
            all_pred.append(pred)
            all_y.append(y)
            
        for batch in val_loader_miss:
            batch = self.process_data(batch)
            y = batch["y"]
            ts_emb = self.ts_encoder(batch)
            pred = self.fusion_model(batch, None, ts_emb)
            pred = torch.sigmoid(pred)
            all_pred_miss.append(pred)
            all_y_miss.append(y)
            all_pred.append(pred)
            all_y.append(y)
            
        all_pred = torch.cat(all_pred, dim=0).cpu().detach().numpy()
        all_y = torch.cat(all_y, dim=0).cpu().detach().numpy()
        all_pred_pair = torch.cat(all_pred_pair, dim=0).cpu().detach().numpy()
        all_y_pair = torch.cat(all_y_pair, dim=0).cpu().detach().numpy()
        all_pred_miss = torch.cat(all_pred_miss, dim=0).cpu().detach().numpy()
        all_y_miss = torch.cat(all_y_miss, dim=0).cpu().detach().numpy()
        
        
        auroc_all = metrics.roc_auc_score(all_y, all_pred)
        auprc_all = metrics.average_precision_score(all_y, all_pred)
        auroc_pair = metrics.roc_auc_score(all_y_pair, all_pred_pair)
        auprc_pair = metrics.average_precision_score(all_y_pair, all_pred_pair)
        auroc_miss = metrics.roc_auc_score(all_y_miss, all_pred_miss)
        auprc_miss = metrics.average_precision_score(all_y_miss, all_pred_miss)
        
        print(f"Epoch {epoch} AUROC_ALL: {auroc_all} AUPRC_ALL: {auprc_all}")
        print(f"Epoch {epoch} AUROC_PAIR: {auroc_pair} AUPRC_PAIR: {auprc_pair}")
        print(f"Epoch {epoch} AUROC_MISS: {auroc_miss} AUPRC_MISS: {auprc_miss}")

        
        return auroc_all, auprc_all, auroc_pair, auprc_pair, auroc_miss, auprc_miss
    
    
    
    def dual_cutoff(self, y_true_val, y_pred_val):
        fpr, tpr, thresholds = metrics.roc_curve(y_true_val, y_pred_val)
        sensitivity = tpr
        specification = 1 - fpr
        lower_cutoff = None
        upper_cutoff = None
        index1 = np.argwhere(sensitivity > 0.9)
        index2 = np.argwhere(specification <= 0.9)

        if len(index1) != 0:
            lower_cutoff = thresholds[index1[0, 0]]
        if len(index2) != 0:
            if index2[0, 0]>0:
                index =index2[0, 0] - 1
                upper_cutoff = thresholds[index]
        else:
            upper_cutoff = thresholds[index2[0, 0]]
            
        return lower_cutoff, upper_cutoff
 
    