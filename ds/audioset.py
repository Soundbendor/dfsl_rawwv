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

#Gemmeke, J. F., Ellis, D. P., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., ... & Ritter, M. (2017, March). Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 776-780). IEEE.

# cur_df should presumably be either the training csv or validation csv
# k is the number of samples per class

# train and valid have 527 distinct classes
class AudioSet(Dataset):
    def __init__(self, cur_df, dset_type="train", classes=list(range(527)), k_shot=80, srate=44100, samp_sz=236196, basefolder = UG.DEF_AUDIOSETDIR, to_label_tx = True, label_offset = 0, one_hot = True, seed = 3):
        self.basepath = basefolder
        self.srate = srate
        # indices
        self.classes = sorted(classes)
        self.labels = self.get_labels()
        self.label_list = sorted(list(self.labels))
        self.audiopath = os.path.join(self.basepath, 'train_wav')
        if dset_type != "train":
            self.audiopath = os.path.join(self.basepath, 'valid_wav')
        self.df = cur_df
        self.num_folds = len(folds)
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
        self.mlb = SKP.MultiLabelBinarizer()
        self.mlb.fit([self.label_list])
        # essentially a way of shuffling
        self.dfsub = self.df.set_index(['fold', 'target']).loc[folds,classes,:].groupby('target').sample(self.k_shot, random_state=seed, replace=False).reset_index()
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

    def get_label_list(self):
        ret = []
        for lgroup in self.cur_df["positive_labels"]:
            cur = [x.strip() for x in lgroup.split(",")]
            ret += cur
        ret = sorted(list(set(ret)))
        return ret

    def get_labels(self):
        ret = {}
        for lgroup in self.cur_df["positive_labels"]:
            cur = [x.strip() for x in lgroup.split(",")]
            for x in cur:
                if x in ret.keys():
                    ret[x] += 1
                else:
                    ret[x] = 1
        ret_ser = pd.Series(data=ret)
        return ret_ser

def make_esc50_fewshot_tasks(cur_df, folds=[], classes=[], n_way = 5, k_shot=np.inf, srate = 16000, samp_sz = 118098, basefolder = UG.DEF_ESC50DIR,, seed = 3, initial_label_offset = 30, to_label_tx = True):
    """
    returns array of (num_classes_added, ds) tups
    """
    ret = []
    num_folds = len(folds)
    num_classes = len(classes)
    num_classes_allocated = 0
    while (num_classes_allocated < num_classes):
        num_classes_to_add = min(n_way, num_classes - num_classes_allocated)
        cur_classes = classes[num_classes_allocated: num_classes_allocated + num_classes_to_add]
        cur_label_offset = initial_label_offset + num_classes_allocated
        cur_ds = ESC50(cur_df, folds=folds, classes=cur_classes, k_shot=k_shot, srate=srate, samp_sz=samp_sz, basefolder = basefolder, seed= seed, label_offset = cur_label_offset, to_label_tx = to_label_tx)
        curtup = (num_classes_to_add, cur_ds)
        ret.append(curtup)
        num_classes_allocated += num_classes_to_add
    return ret
