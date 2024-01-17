import csv
from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
import numpy as np
import os
from util.types import BatchType
import re

def settings_csv_writer(settings_dict, dest_dir="res", expr_idx = 0, epoch_idx=0, expr_name="sampcnn_dfsl"):
    fname = f"{expr_idx}-{expr_name}-settings.csv"
    fpath = os.path.join(dest_dir, fname)
    header = ["sr", "lr", "bs", "epochs", "label_smoothing", "se_dropout", "res1_dropout", "res2_dropout", "rese1_dropout", "rese2_dropout", "simple_dropout", "se_fc_alpha", "rese1_fc_alpha", "rese2_fc_alpha", "use_class_weights"] 
    with open(fpath, "w", newline='', encoding='utf-8') as f:
        dw = csv.DictWriter(f, fieldnames=header)
        dw.writeheader()
        dw.writerow(settings_dict)
        
def res_csv_appender(resdict, dest_dir="res", expr_idx = 0, epoch_idx=0, batch_type=BatchType.train, expr_name="sampcnn_dfsl", pretrain=False):
    fname = f"{expr_idx}-{expr_name}-res.csv"
    if pretrain == True:
        fname = f"{expr_idx}-{expr_name}-res_pretrain.csv"
    fpath = os.path.join(dest_dir, fname)
    header = ["epoch_idx","batch_type","epoch_avg_loss","epoch_avg_time", "epoch_avg_acc1", "epoch_avg_ap"]
    first_write = epoch_idx == 0 and batch_type == BatchType.train
    write_qual = "a" if pretrain == False else "w"
    with open(fpath, write_qual, newline='', encoding='utf-8') as f:
        dw = csv.DictWriter(f, fieldnames=header)
        if first_write == True or pretrain == True:
            dw.writeheader()
        dw.writerow(resdict)

def title_from_key(cur_str):
    return " ".join([x.capitalize() for x in cur_str.split("_")])

def train_valid_grapher(train_arr, valid_arr, dest_dir="graph", graph_key="epoch_avg_loss", expr_idx=0, expr_name="sampcnn_dfsl"):
    gtype = graph_key.split("_")[-1]
    fname = f"{expr_idx}-{expr_name}-{gtype}.png"
    fpath = os.path.join(dest_dir, fname)
    key_title = title_from_key(graph_key)
    ctitle = f"Training and Validation {key_title} for {expr_name}"
    xlabel = "Epoch"
    ylabel = key_title
    plt.suptitle(ctitle)
    train_stuff = [x[graph_key] for x in train_arr]
    valid_stuff = [x[graph_key] for x in valid_arr]
    epochs = list(range(1,len(train_stuff)+1))
    plt.plot(epochs, train_stuff, label="train")
    plt.plot(epochs, valid_stuff, label="valid")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.savefig(fpath)
    plt.clf()



