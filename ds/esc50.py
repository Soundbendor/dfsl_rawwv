import numpy as np
import torch
import os,sys
import pandas as pd
from . import sndload as SL
from torch.utils.data import Dataset
import torch.nn.functional as NF
import torchaudio.transforms as TXT
from sklearn import preprocessing as SKP
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
import util.globals as UG

#Piczak, K.J. (2015). ESC: Dataset for Environmental Sound Classification. In Proceedings of the 23rd ACM International Conference on Multimedia. https://doi.org/10.1145/2733373.2806390
# should be 5 second samples as 44100

# 40 examples total
# 8 examples per fold and 5 folds total

# k is the number of samples per class

class ESC50(Dataset):
    def __init__(self, cur_df, folds=[1,2,3,4,5], classes=list(range(50)), k_shot=80, srate=44100, samp_sz=236196, basefolder = UG.DEF_ESC50DIR, to_label_tx = True, label_offset = 0, one_hot = True, seed = 3):
        self.basepath = basefolder
        self.srate = srate
        self.classes = sorted(classes)
        self.audiopath = os.path.join(self.basepath, 'audio')
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
        self.num_classes_total = self.df['target'].max() + 1 # zero-indexed
        # essentially a way of shuffling
        self.dfsub = self.df.set_index(['fold', 'target']).loc[folds,classes,:].groupby('target').sample(self.k_shot, random_state=seed, replace=False).reset_index()
        self.class_counts = np.array([self.dfsub.loc[self.dfsub['target'] == x].shape[0] for x in self.classes])
        #print(self.dfsub)
        self.samp_sz = samp_sz
        self.shape = self.dfsub.shape
        # remapping a subset of indices to a new set with new offset (for base weightgen training)
        self.subset_remapped_idxs = set([])
        self.subset_is_remapped = False
        self.subset_num_remapped = 0 

    def __len__(self):
        return self.shape[0]

    # unmapped idxs
    def get_class_idxs(self):
        return self.classes
 
    # num_unremapped/remapped overrides class counts for unremapped/remapped
    def get_class_weights(self, num_unremapped = -1, num_remapped = -1):
        tmp = np.zeros(self.num_classes + self.label_offset)
        if self.to_label_tx == True:
            class_counts = self.class_counts
            if num_unremapped > 0:
                class_counts = np.array([1./num_unremapped for _ in self.classes])
            tx_idx = self.label_tx.transform(self.classes) + self.label_offset
            tmp[tx_idx] = class_counts
        if self.subset_is_remapped == True:
            remap_idx = list(self.subset_remapped_idxs)
            tmp[remap_idx] = 0
            tmp2 = np.zeros(self.subset_num_remapped)
            remap_idx2 = self.subset_label_tx(remap_idx)
            tmp2[remap_idxs2] = np.array([self.class_counts[x] for x  in remap_idx])
            if num_remapped > 0:
                tmp2[remap_idxs2] = np.array([1./num_remapped for _  in remap_idx])
            tmp = np.hstack((tmp, tmp2))
        num_classes = np.sum(np.where(tmp > 0., 1., 0.))
        class_prop = np.where( tmp > 0, 1./(tmp*num_classes), 0.)
        #print(class_prop)
        return class_prop

   # given a set of unmapped indices, map them to a new set of indices with offset
    def set_remapped_idx_subset(self, idx_subset):
        self.subset_is_remapped = True
        self.subset_remapped_idxs = set(idx_subset)
        self.subset_label_tx = SKP.LabelEncoder()
        self.subset_label_tx.fit(idx_subset)
        self.subset_label_offset = self.num_classes + self.label_offset
        self.subset_num_remapped = len(idx_subset)

    # unset mapping 
    def unset_remapped_idx_subset(self):
        self.subset_num_remapped = 0 
        del self.subset_label_tx
        self.subset_remapped_idxs = set([])
        self.subset_is_remapped = False


    #unmapped idxs
    def get_class_ex_idxs(self, class_idx):
        return self.dfsub.where(self.dfsub["target"] == class_idx).dropna().index
        
    def get_mapped_class_idxs(self, c_idxs):
        ret_idxs = c_idxs
        if self.to_label_tx == True:
            if self.subset_is_remapped == False:
                ret_idxs = self.label_tx.transform(c_idxs) + self.label_offset
            # mapping subset is activated, check if is in mapped subset or not and act accordingly
            else:
                ret_idxs = []
                for cur_class in c_idxs:
                    cur_ret_idx = 0
                    if cur_class not in self.subset_remapped_idxs:
                        cur_ret_idx = self.label_tx.transform([cur_class])[0] + self.label_offset
                    else:
                        cur_ret_idx = self.subset_label_tx.transform([cur_class])[0] + self.subset_label_offset
                    ret_idxs.append(cur_ret_idx)
        #print(ret_idxs, c_idxs)
        return ret_idxs

     
    def get_mapped_class_idx(self, c_idx):
        ret_idx = c_idx
        if self.to_label_tx == True:
            if self.subset_is_remapped == False:
                ret_idx = self.label_tx.transform([c_idx])[0] + self.label_offset
            # mapping subset is activated, check if is in mapped subset or not and act accordingly
            elif c_idx not in self.subset_remapped_idxs:
                ret_idx = self.label_tx.transform([c_idx])[0] + self.label_offset
            else:
                ret_idx = self.subset_label_tx.transform([c_idx])[0] + self.subset_label_offset

 
        return ret_idx


    def __getitem__(self, idx):
        # return sound, label
        cur_entry = self.dfsub.iloc[idx]
        cur_fpath = os.path.join(self.audiopath, cur_entry['filename'])
        cur_class = cur_entry['target']
        ret_label = None
        if self.to_label_tx == True:
            if self.subset_is_remapped == False:
                cur_class = self.label_tx.transform([cur_class])[0] + self.label_offset
            # mapping subset is activated, check if is in mapped subset or not and act accordingly
            elif cur_class not in self.subset_remapped_idxs:
                cur_class = self.label_tx.transform([cur_class])[0] + self.label_offset
            else:
                cur_class = self.subset_label_tx.transform([cur_class])[0] + self.subset_label_offset

        if self.one_hot == True:
            if self.subset_is_remapped == False:
                ret_label = NF.one_hot(torch.tensor(cur_class), num_classes=self.num_classes)
            else:
                ret_label = NF.one_hot(torch.tensor(cur_class), num_classes=(self.num_classes +  self.subset_num_remapped))
        else:
            ret_label = cur_class
        cur_snd = SL.sndloader(cur_fpath, want_sr=self.srate, max_samp=self.samp_sz, to_mono=True)
        return cur_snd, ret_label


def make_esc50_fewshot_tasks(cur_df, folds=[], classes=[], n_way = 5, k_shot=np.inf, srate = 16000, samp_sz = 118098, basefolder = UG.DEF_ESC50DIR, seed = 3, one_hot = True, initial_label_offset = 30, to_label_tx = True):
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
        cur_ds = ESC50(cur_df, folds=folds, classes=cur_classes, k_shot=k_shot, srate=srate, samp_sz=samp_sz, basefolder = basefolder, seed= seed, one_hot = one_hot, label_offset = cur_label_offset, to_label_tx = to_label_tx)
        curtup = (num_classes_to_add, cur_ds)
        ret.append(curtup)
        num_classes_allocated += num_classes_to_add
        #print(classes)
    return ret
