import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms



class Multi_Dataset_Pair(Dataset):
    def __init__(self, type="train", pair=True):
        super(Multi_Dataset_Pair, self).__init__()
        
        self.patient = np.array("./all_patient_pair_{}.npy".format(type))

        self.p_id = self.patient[0]
        self.p_label = self.patient[1]
        self.type = type

        self.ct = []
        self.ct_t = []
        self.y = []
        
        for i, id in self.p_id:
            ct_seq = np.load("./data/ts/{}.npy".format(id))
            for i in range(len(ct_seq)):
                self.ct.append(ct_seq[i][0])
                self.ct_t.append(ct_seq[i][1]) 
                self.y.append(self.p_label[id])
        
        ts_data = np.load("./data/ts/all_ts.npy")
        self.ts = ts_data[0]
        self.mask = ts_data[1]
        self.ts_t = ts_data[2]             
                   
    def __len__(self):
        return len(self.p_id) 

    def __getitem__(self, idx):
        id = self.p_id[idx]
        img = self.ct[idx]
        ct_t = self.ct_t[idx]
        y = self.y[idx]
        
        ts = self.ts[id]
        mask = self.mask[id]
        ts_t = self.ts_t[id]
        
        data = {
            "ct": img,
            "ct_t": ct_t,
            "ts": ts,
            "ts_t": ts_t,
            "mask": mask,
            "label": y
            }
        return data



class Multi_Dataset_Miss(Dataset):
    def __init__(self, type="train", pair=True):
        super(Multi_Dataset_Miss, self).__init__()
        
        self.patient = np.array("./all_patient_miss_{}.npy".format(type))

        self.p_id = self.patient[0]
        self.p_label = self.patient[1]
        self.type = type


        ts_data = np.load("./data/ts/all_ts.npy")
        self.ts = ts_data[0]
        self.mask = ts_data[1]
        self.ts_t = ts_data[2]             
                   
    def __len__(self):
        return len(self.p_id) 

    def __getitem__(self, idx):
        id = self.p_id[idx]
        y  = self.p_label[id]
        
        ts = self.ts[id]
        mask = self.mask[id]
        ts_t = self.ts_t[id]
        
        data = {
            "ts": ts,
            "ts_t": ts_t,
            "mask": mask,
            "label": y
            }
        return data




class collater():
    def __init__(self):
        self.test = None
        self.keys = ["ct"]
    def __call__(self, batch):
        all_ct = []
        all_ts = []
        all_mask = []
        all_ts_t = []
        ct_idx = []
        all_ct_t = []
        all_y = []
        
        for i, data in enumerate(batch):
            all_ct.append(data["ct"])
            all_ct_t.append(data["ct_t"])
            for j in range(len(data["ct"])):
                ct_idx.append(i + 1)
            all_ts.append(data["ts"]) 
            all_ts_t.append(data["ts_t"])
            all_mask.append(data["mask"]) 
            all_y.append(data["label"])  
        dicts = {
            "ct": torch.cat(all_ct, dim=0),
            "ct_t": torch.cat(all_ct_t, dim=0),
            "ct_idx": torch.LongTensor(ct_idx),
            "ts": torch.cat(all_ts, dim=0),
            "ts_t": torch.cat(all_ts_t, dim=0),
            "mask": torch.cat(all_mask, dim=0),
            "y": torch.tensor(all_y, dtype=torch.float32)
        }
        
        
        return dicts



def get_multi_dataloader(batch_size):


    collate_fn = collater()
    
    train_set_pair = Multi_Dataset_Pair(type="train")
    val_set_pair = Multi_Dataset_Pair(type="val")
    test_set_pair = Multi_Dataset_Pair(type="test")
    
    train_set_miss = Multi_Dataset_Miss(type="train")
    val_set_miss = Multi_Dataset_Miss(type="val")
    test_set_miss = Multi_Dataset_Miss(type="test")
    
    
    train_loader_pair = DataLoader(
        train_set_pair, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    val_loader_pair = DataLoader(
        val_set_pair, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader_pair = DataLoader(
        test_set_pair, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    train_loader_miss = DataLoader(train_set_miss, batch_size=batch_size, shuffle=True)
    val_loader_miss = DataLoader(val_set_miss, batch_size=batch_size, shuffle=False)
    test_loader_miss = DataLoader(test_set_miss, batch_size=batch_size, shuffle=False)
    
    return train_loader_pair, val_loader_pair, test_loader_pair, train_loader_miss, val_loader_miss, test_loader_miss



