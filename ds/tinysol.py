import numpy as np
import torch
import os
import pandas as pd
from . import sndload as SL
from torch.utils.data import Dataset
import torch.nn.functional as NF
import torchaudio.transforms as TXT
from sklearn import preprocessing as SKP
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
import util.globals as UG

#Carmine Emanuele, Daniele Ghisi, Vincent Lostanlen, Fabien Lévy, Joshua Fineberg, & Yan Maresz. (2020). TinySOL: an audio dataset of isolated musical notes (6.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3685367
# should be 5 second samples as 44100

#totalsize: 1532
# instrument breakdown

#Vc: 156,Va: 152,Vn: 147,Cb: 144,Hn: 134,Bn: 126,ClBb: 126,Tbn: 119,Fl: 118,BTb: 107,Ob: 107,TpC: 96

#dynamic breakdown: mf: 511, ff: 510, pp: 509, p: 2

# most instruments have 3 examples per note, sommetimes there are less than < 3 examples per note (as low as 1), sometimes there are as much as 6
class TinySOL(Dataset):
    def __init__(self, cur_df, classes=list(range(12)), k_shot=80, srate=44100, samp_sz=236196, basefolder = UG.DEF_TINYSOLDIR, to_label_tx = True, label_offset = 0, one_hot = True, seed = 3):
        self.inst_list = ['ClBb', 'Vn', 'Ob', 'Va', 'Fl', 'BTb', 'Cb', 'Bn', 'Tbn', 'TpC', 'Hn', 'Vc']
        self.inst_to_idx = {x:inst_list.index(x) for x in inst_list}
        self.idx_to_inst = {inst_list.index(x):x for x in inst_list}
        self.idx_list = [x for x in range(len(self.inst_list))]
        self.basepath = basefolder
        self.srate = srate
        # classes are indices to keep compatibility with esc50
        self.classes = sorted(classes)
        # so set as a string
        self.classes_str = [self.inst_list[x] for x in self.classes]
        self.df = cur_df
        # 5 folds, max 8 samples per fold
        # k = number of examples per class
        self.k_shot = max(1,min(k_shot, 8 * self.num_folds))
        self.num_classes = len(self.classes)
        self.to_label_tx = to_label_tx
        if to_label_tx == True:
            self.label_tx = SKP.LabelEncoder()
            self.label_tx.fit(self.classes)
            self.label_offset = label_offset
        self.one_hot = one_hot
        self.num_classes_total = len(self.inst_list)
        # essentially a way of shuffling
        self.dfsub = self.df.set_index(['instrument']).loc[self.classes_str,:].groupby('instrument').sample(self.k_shot, random_state=seed, replace=False).reset_index()
        #print(self.dfsub)
        self.samp_sz = samp_sz
        self.shape = self.dfsub.shape
         
    def __len__(self):
        return self.shape[0]

    # unmapped idxs
    def get_class_idxs(self):
        return self.classes
    
    #unmapped idxs
    def get_class_ex_idxs(self, class_idx):
        class_str = self.inst_list[class_idx]
        return self.dfsub.where(self.dfsub["instrument"] == class_str).dropna().index
        
    def get_mapped_class_idxs(self, c_idxs):
        ret_idxs = c_idxs
        if self.to_label_tx == True:
            ret_idxs = self.label_tx.transform(c_idxs) + self.label_offset
        return ret_idxs

     
    def get_mapped_class_idx(self, c_idx):
        ret_idx = c_idx
        if self.to_label_tx == True:
            ret_idx = self.label_tx.transform([c_idx])[0] + self.label_offset
        return ret_idx


    def __getitem__(self, idx):
        # return sound, label
        cur_entry = self.dfsub.iloc[idx]
        cur_fpath = os.path.join(self.basepath, cur_entry['path'])
        # class as string
        cur_class_str = cur_entry['instrument']
        # class as int
        cur_class = self.inst_to_idx[cur_class_str]
        ret_label = None
        if self.to_label_tx == True:
            cur_class = self.label_tx.transform([cur_class])[0] + self.label_offset
        if self.one_hot == True:
            ret_label = NF.one_hot(torch.tensor(cur_class), num_classes=self.num_classes)
        else:
            ret_label = cur_class
        cur_snd = SL.sndloader(cur_fpath, want_sr=self.srate, max_samp=self.samp_sz, to_mono=True)
        return cur_snd, ret_label


def make_tinysol_fewshot_tasks(cur_df, classes=[], n_way = 5, k_shot=np.inf, srate = 16000, samp_sz = 118098,  basefolder = UG.DEF_TINYSOLDIR, seed = 3, initial_label_offset = 30, to_label_tx = True):
    """
    returns array of (num_classes_added, ds) tups
    """
    ret = []
    num_classes = len(classes)
    num_classes_allocated = 0
    while (num_classes_allocated < num_classes):
        num_classes_to_add = min(n_way, num_classes - num_classes_allocated)
        cur_classes = classes[num_classes_allocated: num_classes_allocated + num_classes_to_add]
        cur_label_offset = initial_label_offset + num_classes_allocated
        cur_ds = TinySOL(cur_df, classes=cur_classes, k_shot=k_shot, srate=srate, samp_sz=samp_sz, basefolder = basefolder, seed= seed, label_offset = cur_label_offset, to_label_tx = to_label_tx)
        curtup = (num_classes_to_add, cur_ds)
        ret.append(curtup)
        num_classes_allocated += num_classes_to_add
 
