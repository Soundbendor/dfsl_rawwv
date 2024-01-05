import numpy as np
import torch
import os
import pandas as pd
from . import sndload as SL
from torch.utils.data import Dataset
import torch.nn.functional as NF
import torchaudio.transforms as TXT

#Piczak, K.J. (2015). ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd ACM International Conference on Multimedia. https://doi.org/10.1145/2733373.2806390
# should be 5 second samples as 44100
class ESC50(Dataset):
    def __init__(self, folds=[1,2,3,4,5], classes=list(range(50)), k=80, srate=44100, samp_sz=236196, basefolder = os.path.join(os.path.split(__file__)[0], "ESC-50-master"), seed = 3):
        self.basepath = basefolder
        self.srate = srate
        self.audiopath = os.path.join(self.basepath, 'audio')
        self.csvpath = os.path.join(self.basepath,"meta", "esc50.csv")
        self.df = pd.read_csv(self.csvpath)
        # 5 folds, max 8 samples per fold
        self.k = max(1,min(k, 8 * len(folds)))
        self.num_classes = self.df['target'].max() + 1 # zero-indexed
        # k = number of instances per class
        self.dfsub = self.df.set_index(['fold', 'target']).loc[folds,classes,:].groupby('target').sample(self.k, random_state=seed).reset_index()
        self.samp_sz = samp_sz
        self.shape = self.dfsub.shape
         
    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        # return sound, label
        cur_entry = self.dfsub.iloc[idx]
        cur_fpath = os.path.join(self.audiopath, cur_entry['filename'])
        cur_class = cur_entry['target']
        cur_onehot = NF.one_hot(torch.tensor(cur_class), num_classes=self.num_classes)
        cur_snd = SL.sndloader(cur_fpath, want_sr=self.srate, max_samp=self.samp_sz, to_mono=True)
        return cur_snd, cur_onehot


