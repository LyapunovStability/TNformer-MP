import argparse
import torch
import datetime
import json
import yaml
import os
from trainer import Trainer

parser = argparse.ArgumentParser(description="Multi_Modal")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument('--device', default='cuda:0', help='Device')


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
args = parser.parse_args()

device = args.device


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)
print(json.dumps(config, indent=4))
print(args)

config["train"]["device"] = args.device


foldername = "./save/{}" + "_" + "multi_modal" + "_" + current_time.format(config["train"]["dataset"]) + "/"

print('model folder:', foldername)
if foldername != "":
    os.makedirs(foldername, exist_ok=True)


trainer = Trainer(config)

# start training
Trainer.uni_modal_train_ct()
Trainer.uni_modal_train_ts()
Trainer.multi_modal_train()







