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

#Carmine Emanuele, Daniele Ghisi, Vincent Lostanlen, Fabien Lévy, Joshua Fineberg, & Yan Maresz. (2020). TinySOL: an audio dataset of isolated musical notes (6.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3685367
# should be 5 second samples as 44100

#totalsize: 2913
# instrument breakdown
# Acc     689
#Cb      309
#Va      309
#Vc      291
#Vn      284
#Hn      134
#ClBb    126
#Bn      126
#Fl      118
#Tbn     117
#BTb     108
#Ob      107
#ASax     99
#TpC      96

# max samples per inst per fold = 138
# min samples per inst per fold = 19
#14 classes

# lengths can be 2 to 10 seconds, maybe pick 5 as a nice middle ground
# most instruments have 3 examples per note, sommetimes there are less than < 3 examples per note (as low as 1), sometimes there are as much as 6
class TinySOL(Dataset):
    def __init__(self, cur_df, classes=list(range(12)), folds =list(range(5)), k_shot=80, srate=44100, samp_sz=236196, basefolder = UG.DEF_TINYSOLDIR, to_label_tx = True, label_offset = 0, one_hot = True, seed = 3):
        #self.inst_list = ['ClBb', 'Vn', 'Ob', 'Va', 'Fl', 'BTb', 'Cb', 'Bn', 'Tbn', 'TpC', 'Hn', 'Vc']
        self.inst_list = ['Acc', 'Cb', 'Va', 'Vc', 'Vn', 'Hn', 'ClBb', 'Bn', 'Fl', 'Tbn', 'BTb', 'Ob', 'ASax', 'TpC'] 
        self.inst_to_idx = {x:self.inst_list.index(x) for x in self.inst_list}
        self.idx_to_inst = {self.inst_list.index(x):x for x in self.inst_list}
        self.idx_list = [x for x in range(len(self.inst_list))]
        self.folds = sorted(folds)
        self.num_folds = len(self.folds)
        self.basepath = basefolder
        self.srate = srate
        self.inst_cat = 'Instrument (abbr.)'
        self.fold_cat = 'Fold'
        self.path_cat = 'Path'
        # classes are indices to keep compatibility with esc50
        self.classes = sorted(classes)
        # so set as a string
        self.classes_str = [self.inst_list[x] for x in self.classes]
        self.df = cur_df
        # 5 folds, max 8 samples per fold
        # k = number of examples per class
        self.k_shot = max(1,min(k_shot, 138 * self.num_folds))
        self.num_classes = len(self.classes)
        self.to_label_tx = to_label_tx
        if to_label_tx == True:
            self.label_tx = SKP.LabelEncoder()
            self.label_tx.fit(self.classes)
            self.label_offset = label_offset
        self.one_hot = one_hot
        self.num_classes_total = len(self.inst_list)
        # essentially a way of shuffling
        #self.dfsub = self.df.set_index(['fold', 'target']).loc[folds,classes,:].groupby('target').sample(self.k_shot, random_state=seed, replace=False).reset_index()
        self.dfsub = self.df.set_index([self.fold_cat, self.inst_cat]).loc[self.folds, self.classes_str,:].groupby(self.inst_cat).sample(self.k_shot, random_state=seed, replace=False).reset_index()
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
    
    #unmapped idxs
    def get_class_ex_idxs(self, class_idx):
        class_str = self.inst_list[class_idx]
        return self.dfsub[self.dfsub[self.inst_cat] == class_str].index
        
    
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
        self.subset_mapped_idxs = set([])
        self.subset_is_remapped = False


    def __getitem__(self, idx):
        # return sound, label
        cur_entry = self.dfsub.iloc[idx]
        cur_fpath = os.path.join(self.basepath, cur_entry[self.path_cat])
        # class as string
        cur_class_str = cur_entry[self.inst_cat]
        # class as int
        cur_class = self.inst_to_idx[cur_class_str]
        ret_label = None
        if self.to_label_tx == True:
            if self.subset_is_remapped == False:
                cur_class = self.label_tx.transform([cur_class])[0] + self.label_offset
            # mapping subset is activated, check if is in mapped subset or not and act accordingly
            elif cur_class not in self.subset_mapped_idxs:
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


def make_tinysol_fewshot_tasks(cur_df, folds=[], classes=[], n_way = 5, k_shot=np.inf, srate = 16000, samp_sz = 118098,  basefolder = UG.DEF_TINYSOLDIR, seed = 3, initial_label_offset = 30, one_hot = True, to_label_tx = True):
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
        cur_ds = TinySOL(cur_df, folds=folds, classes=cur_classes, k_shot=k_shot, srate=srate, samp_sz=samp_sz, basefolder = basefolder, seed= seed, label_offset = cur_label_offset, one_hot = one_hot, to_label_tx = to_label_tx)
        curtup = (num_classes_to_add, cur_ds)
        ret.append(curtup)
        num_classes_allocated += num_classes_to_add
    return ret 
