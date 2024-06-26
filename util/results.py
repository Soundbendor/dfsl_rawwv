import csv
from matplotlib import pyplot as plt
from matplotlib import ticker as tkr
import numpy as np
import os
from util.types import BatchType,TrainPhase,ModelName,DatasetName
import re
import util.metrics as UM
import neptune

def settings_csv_writer(settings_dict, dest_dir="res", expr_num = 0, epoch_idx=0, modelname=ModelName.samplecnn, baseset=DatasetName.esc50, novelset=DatasetName.esc50, expr_name="sampcnn_dfsl"):
    fname = f"{expr_num}-{modelname.name}-{baseset.name}-{novelset.name}_settings.csv"
    fpath = os.path.join(dest_dir, fname)
    header = ["expr_num", "sr", "lr", "bs", "epochs", "label_smoothing", "se_dropout", "res1_dropout", "res2_dropout", "rese1_dropout", "rese2_dropout", "simple_dropout", "se_fc_alpha", "rese1_fc_alpha", "rese2_fc_alpha", "use_class_weights", "omit_last_relu", "use_prelu", "se_prelu", "multilabel", 'cls_fn'] 
    with open(fpath, "w", newline='', encoding='utf-8') as f:
        dw = csv.DictWriter(f, fieldnames=header)
        dw.writeheader()
        dw.writerow(settings_dict)
        
def res_csv_appender(resdict, dest_dir="res", expr_num = 0, epoch_idx=0, batch_type=BatchType.train, modelname=ModelName.samplecnn, baseset=DatasetName.esc50, novelset=DatasetName.esc50,train_phase=TrainPhase.base_init, pretrain=False):
    fname = f"{expr_num}-{modelname.name}-{baseset.name}-{novelset.name}_{train_phase.name}-res.csv"
    if pretrain == True:
        fname = f"{expr_num}-{modelname.name}-{baseset.name}-{novelset.name}_res-{train_phase.name}-pretrain.csv"
    fpath = os.path.join(dest_dir, fname)
    header = ["epoch_idx","dataset", "ds_idx", "batch_type","loss_avg","time_avg"]
    if resdict["multilabel"] == False:
        header += UM.csvable_mc
    else:
        header += UM.csvable_ml
    first_write = epoch_idx == 0 and batch_type == BatchType.train
    write_qual = "a" if pretrain == False else "w"
    with open(fpath, write_qual, newline='', encoding='utf-8') as f:
        dw = csv.DictWriter(f, fieldnames=header)
        if first_write == True or pretrain == True:
            dw.writeheader()
        dw.writerow({k:v for k,v in resdict.items() if k in header})

def title_from_key(cur_str):
    return " ".join([x.capitalize() for x in cur_str.split("_")])

def train_valid_grapher(train_arr, valid_arr, dest_dir="graph", graph_key="loss_avg", expr_num=0,modelname=ModelName.samplecnn, baseset=DatasetName.esc50, novelset=DatasetName.esc50):
    gtype = graph_key.split("_")[-1]
    fname = f"{expr_num}-{modelname.name}-{baseset.name}-{novelset.name}_{gtype}.png"
    fpath = os.path.join(dest_dir, fname)
    key_title = title_from_key(graph_key)
    ctitle = f"Training and Validation {key_title}\n for {modelname.name}:{baseset.name}-{novelset.name}"
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

def plot_confmat(confmat,dest_dir="graph", confmat2 = None, train_phase = TrainPhase.base_init, expr_num=0, modelname=ModelName.samplecnn, baseset=DatasetName.esc50, novelset=DatasetName.esc50, is_base = True, multilabel=False):

    t_ph_name = train_phase.name
    t_ph_title = title_from_key(t_ph_name)
    addstr = "base"
    paddstr = "Base Set"
    if is_base == False:
        addstr = "novel"
        paddstr  = "Novel Set"
    if confmat2 != None:
        addstr = "basenovel"
        paddstr = "Base+Novel Set"
    fname = f"{expr_num}-{addstr}-{modelname.name}-{baseset.name}-{novelset.name}_{train_phase.name}-confmat.png"
    ctitle = f"{t_ph_title} ({paddstr}) Confusion Matrix\nfor {modelname.name}:{baseset.name}-{novelset.name}"
    fpath = os.path.join(dest_dir, fname)
    if multilabel == False:
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        confmat_show = confmat
        if confmat2 != None:
            confmat_show = confmat + confmat2
        cur=ax.imshow(confmat_show.cpu(),cmap='BuPu')
        row=confmat.shape[0]
        majorstep = 10
        minorstep=1
        majortix=np.arange(0,row,majorstep)
        minortix=np.arange(-0.5,row,minorstep)
        ax.tick_params(labelbottom=False,which="major",bottom=False,top=False,labeltop=True,left=False, labelleft=True, labelright=False,right=False)
        ax.tick_params(labelbottom=False,which="minor",bottom=False,top=True,labeltop=False,left=True, labelleft=False, labelright=False,right=False)
        #plt.colorbar(cur)
        ax.set_xticks(majortix)
        ax.set_xticks(minortix, minor=True)
        ax.set_yticks(majortix)
        ax.set_yticks(minortix, minor=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        #ax.grid(visible=True,which="minor", color="white", linestyle="-", alpha=0.5,linewidth=1)
    else:
        fig,ax = confmat.plot()
    plt.suptitle(ctitle)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.clf()
    return fpath



