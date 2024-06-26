import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms



class CT_Dataset(Dataset):
    def __init__(self, type="train", transform=None):
        super(CT_Dataset, self).__init__()
        self.patient = np.array("./all_patient_{}.npy".format(type))
        self.p_id = self.patient[0]
        self.p_label = self.patient[1]
        self.type = type
        self.transform = transform
        self.ct = []
        self.ct_t = []
        self.y = []
        for i, id in self.p_id:
            ct_seq = np.load("./data/ts/{}.npy".format(id))
            for i in range(len(ct_seq)):
                self.ct.append(ct_seq[i][0])
                self.ct_t.append(ct_seq[i][1]) 
                self.y.append(self.p_label[id])
                       
    def __len__(self):
        return len(self.ct) 

    def __getitem__(self, idx):
        img = self.ct[idx]
        t = self.ct_t[idx]
        y = self.y[idx]
        img_transformed = self.transform(img)
        s = {
            "ct": img_transformed,
            "ct_t": t,
            "y": y
            }
        return s


class TS_Dataset(Dataset):
    def __init__(self, type="train"):
        super(TS_Dataset, self).__init__()
        patient = np.array("./all_patient_{}.npy".format(type))
        self.p_id = patient[0]
        self.p_label = patient[1]
        self.type = type
        
        ts_data = np.load("./data/ts/all_ts.npy")
        self.ts = ts_data[0]
        self.mask = ts_data[1]
        self.t = ts_data[2]
        

    def __len__(self):
        return len(self.p_id) 

    def __getitem__(self, idx):
        id = self.p_id[idx]
        
        ts = self.ts[id]
        mask = self.mask[id]
        y = self.p_label[id]
        t = self.t[id]
        s = {
            "ts": ts,
            "mask": mask,
            "ts_t": t,
            "y": y
            }
        return s




class collater():
    def __init__(self):
        self.test = None
        self.keys = ["ct"]
    def __call__(self, batch):
        all_ct = []
        ct_idx = []
        all_ct_t = []
        all_y = []
        
        for i, data in enumerate(batch):
            all_ct.append(data["ct"])
            all_ct_t.append(data["ct_t"])
            for j in range(len(data["ct"])):
                ct_idx.append(i + 1)  
            all_y.append(data["label"])  
        dicts = {
            "ct": torch.cat(all_ct, dim=0),
            "ct_t": torch.cat(all_ct_t, dim=0),
            "ct_idx": torch.LongTensor(ct_idx),
            "label": torch.tensor(all_y, dtype=torch.float32)
        }
        
        
        return dicts



def get_img_dataloader(batch_size=16):

    train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    )

    val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    )


    test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    )

    train_set = CT_Dataset(type="train", transform=train_transforms)
    val_set = CT_Dataset(type="val", transform=val_transforms)
    test_set = CT_Dataset(type="test", transform=test_transforms)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_ts_dataloader(batch_size=32):

    train_set = TS_Dataset(type="train")
    val_set = TS_Dataset(type="val")
    test_set = TS_Dataset(type="test")
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

