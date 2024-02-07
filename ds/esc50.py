import numpy as np
import torch
import os
import pandas as pd
from . import sndload as SL
from torch.utils.data import Dataset
import torch.nn.functional as NF
import torchaudio.transforms as TXT
from sklearn import preprocessing as SKP

#Piczak, K.J. (2015). ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd ACM International Conference on Multimedia. https://doi.org/10.1145/2733373.2806390
# should be 5 second samples as 44100

# 40 examples total
# 8 examples per fold and 5 folds total
class ESC50(Dataset):
    def __init__(self, folds=[1,2,3,4,5], classes=list(range(50)), k=80, srate=44100, samp_sz=236196, basefolder = os.path.join(os.path.split(__file__)[0], "ESC-50-master"), to_label_tx = True, label_offset = 0, one_hot = True, seed = 3):
        self.basepath = basefolder
        self.srate = srate
        self.classes = sorted(classes)
        self.audiopath = os.path.join(self.basepath, 'audio')
        self.csvpath = os.path.join(self.basepath,"meta", "esc50.csv")
        self.df = pd.read_csv(self.csvpath)
        # 5 folds, max 8 samples per fold
        self.k = max(1,min(k, 8 * len(folds)))
        self.num_classes = len(classes)
        self.to_label_tx = to_label_tx
        if to_label_tx == True:
            self.label_tx = SKP.LabelEncoder()
            self.label_tx.fit(classes)
            self.label_offset = label_offset
        self.one_hot = one_hot
        self.num_classes_total = self.df['target'].max() + 1 # zero-indexed
        self.class_counts = np.array([self.df.loc[self.df['target'] == x].shape[0] for x in self.classes])
        self.class_prop = 1./self.class_counts
        # k = number of instances per class
        self.dfsub = self.df.set_index(['fold', 'target']).loc[folds,classes,:].groupby('target').sample(self.k, random_state=seed).reset_index()
        self.samp_sz = samp_sz
        self.shape = self.dfsub.shape
         
    def __len__(self):
        return self.shape[0]

    def get_class_idxs(self):
        return self.classes

    def get_class_ex_idxs(self, class_idx):
        return self.dfsub.where(self.dfsub["target"] == class_idx).dropna().index
        
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
        cur_fpath = os.path.join(self.audiopath, cur_entry['filename'])
        cur_class = cur_entry['target']
        ret_label = None
        if self.to_label_tx == True:
            cur_class = self.label_tx.transform([cur_class])[0] + self.label_offset
        if self.one_hot == True:
            ret_label = NF.one_hot(torch.tensor(cur_class), num_classes=self.num_classes)
        else:
            ret_label = cur_class
        cur_snd = SL.sndloader(cur_fpath, want_sr=self.srate, max_samp=self.samp_sz, to_mono=True)
        return cur_snd, ret_label


