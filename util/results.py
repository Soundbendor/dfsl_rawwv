import csv
from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
import numpy as np
import os
from util.types import BatchType

def settings_csv_writer(settings_dict, dest_dir="res", expr_idx = 0, epoch_idx=0, expr_name="sampcnn_dfsl"):
    fname = f"{expr_idx}-{expr_name}-settings.csv"
    fpath = os.path.join(dest_dir, fname)
    header = ["lr", "bs", "epochs"] 
    with open(fpath, "w", newline='', encoding='utf-8') as f:
        dw = csv.DictWriter(f, fieldnames=header)
        dw.writeheader()
        dw.writerow(settings_dict)
        
def res_csv_appender(resdict, dest_dir="res", expr_idx = 0, epoch_idx=0, batch_type=BatchType.train, expr_name="sampcnn_dfsl", pretrain=False):
    fname = f"{expr_idx}-{expr_name}-res.csv"
    if pretrain == True:
        fname = f"{expr_idx}-{expr_name}-res_pretrain.csv"
    fpath = os.path.join(dest_dir, fname)
    header = ["epoch_idx","batch_type","batch_avg_loss","batch_avg_time"]
    first_write = epoch_idx == 0 and batch_type == BatchType.train
    write_qual = "a" if pretrain == False else "w"
    with open(fpath, write_qual, newline='', encoding='utf-8') as f:
        dw = csv.DictWriter(f, fieldnames=header)
        if first_write == True or pretrain == True:
            dw.writeheader()
        dw.writerow(resdict)
        
def train_valid_loss_grapher(train_arr, valid_arr, dest_dir="graph", expr_idx=0, expr_name="sampcnn_dfsl"):
    fname = f"{expr_idx}-{expr_name}-res.png"
    fpath = os.path.join(dest_dir, fname)
    ctitle = f"Training and Validation Loss for {expr_name}"
    xlabel = "Epoch"
    ylabel = "Loss"
    plt.suptitle(ctitle)
    train_losses = [x["batch_avg_loss"] for x in train_arr]
    valid_losses = [x["batch_avg_loss"] for x in valid_arr]
    epochs = list(range(len(train_losses)))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, valid_losses, label="valid")
    plt.legend(loc="upper right")
    plt.savefig(fpath)
    plt.clf()



