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

#S. Wang, A. Mesaros, T. Heittola and T. Virtanen, "A Curated Dataset of Urban Scenes for Audio-Visual Scene Analysis," IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Toronto, ON, Canada, 2021, pp. 626-630, doi: 10.1109/ICASSP39728.2021.9415085.

# loc: Amsterdam, Barcelona, Helsinki, Lisbon, London, Lyon, Madrid, Milan, Prague, Paris, Stockholm, and Vienna.

# classes
# airport, shopping mall (indoor), metro station (underground), pedestrian street, public square, street (traffic), traveling by tram, bus and metro (underground), and urban park.

# test dataframe does not have labels (just filename)
# same with openset eval and leaderboard

# all 10 seconds long so maybe index so you take the first 5 and then the last 5

# note that dev set is 48 kHz stereo and eval sets are 44.1 kHz mono


#train (fold1_train) - 9185, around 900 per class
#valid (fold1_evaluate), 4185, around 400 per class
#test (fold1_testwlabels, custom merge of meta.csv and fol1_test) - 4185, around 400 per class


class TAU(Dataset):
    def __init__(self, cur_df, classes=list(range(10)), k_shot=80, srate=44100, samp_sz=236196, basefolder = UG.DEF_TAUDEVDIR, to_label_tx = True, label_offset = 0, one_hot = True, is_dev = True, seed = 3):
        #self.inst_list = ['ClBb', 'Vn', 'Ob', 'Va', 'Fl', 'BTb', 'Cb', 'Bn', 'Tbn', 'TpC', 'Hn', 'Vc']
        self.classlist = ['airport', 'bus', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'metro_station', 'park', 'public_square', 'metro', 'tram']
        self.cls_to_idx = {x:self.classlist.index(x) for x in self.classlist}
        self.idx_to_cls = {self.classlist.index(x):x for x in self.classlist}
        self.idx_list = [x for x in range(len(self.classlist))]
        #self.folds = sorted(folds)
        #self.num_folds = len(self.folds)
        self.basepath = basefolder
        self.srate = srate
        self.class_cat = 'scene_label'
        self.path_cat = 'filename'
        # classes are indices to keep compatibility with esc50
        self.classes = sorted(classes)
        # so set as a string
        self.classes_str = [self.classlist[x] for x in self.classes]
        self.df = cur_df
        self.num_folds = 1 # no folds
        self.div = 2 # number of subsamples to divide each sample into
        #self.k_shot = max(1,min(k_shot, 1440 * self.div * self.num_folds))
        self.k_shot = max(1,k_shot)
        self.num_classes = len(self.classes)
        self.to_label_tx = to_label_tx
        if to_label_tx == True:
            self.label_tx = SKP.LabelEncoder()
            self.label_tx.fit(self.classes)
            self.label_offset = label_offset
        self.one_hot = one_hot
        self.num_classes_total = len(self.classlist)
        # essentially a way of shuffling
        #self.dfsub = self.df.set_index(['fold', 'target']).loc[folds,classes,:].groupby('target').sample(self.k_shot, random_state=seed, replace=False).reset_index()
        if np.isfinite(self.k_shot) == True:
            self.dfsub = self.df.set_index([self.class_cat]).loc[self.classes_str,:].groupby(self.class_cat).sample(self.k_shot//2, random_state=seed, replace=False).reset_index()
        else:
            self.dfsub = self.df.query(f'{self.class_cat} in {self.classes_str}').reset_index()
        self.class_counts = np.array([self.dfsub.loc[self.dfsub[self.class_cat] == x].shape[0] * self.div for x in self.classes_str])
        #print(self.dfsub)
        self.samp_sz = samp_sz
        self.shape = self.dfsub.shape
        # remapping a subset of indices to a new set with new offset (for base weightgen training)
        self.subset_remapped_idxs = set([])
        # halfway sample

        # each file is 10 seconds long
        if is_dev == True:
            self.samp_offset = 48000 * int(10./(self.div))
        else:
            self.samp_offset = 44100 * int(10./(self.div))
        self.subset_is_remapped = False
        self.subset_num_remapped = 0 
         
    def __len__(self):
        return self.shape[0] * self.div

    # unmapped idxs
    def get_class_idxs(self):
        return self.classes
    
    #unmapped idxs
    def get_class_ex_idxs(self, class_idx):
        # needs to take into consideration dividing samples
        class_str = self.classlist[class_idx]
        idxs = self.dfsub[self.dfsub[self.class_cat] == class_str].index * self.div
        ret = idxs
        if self.div > 1:
            for i in range(self.div-1):
                # i is 0 and want to start with 1
                cur = idxs + (i + 1)
                ret = np.hstack((ret, cur))
        return ret

        
    
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
        self.subset_remapped_idxs = set([])
        self.subset_is_remapped = False

    # a simple warapper around indexing and offsetting
    def idx_map(self,in_idx):
        cur_idx = int(in_idx/self.div)
        cur_offset = int((in_idx % self.div) * self.samp_offset)
        return cur_idx, cur_offset




    def __getitem__(self, idx):
        # return sound, label
        cur_idx, cur_offset = self.idx_map(idx)
        cur_entry = self.dfsub.iloc[cur_idx]
        cur_fpath = os.path.join(self.basepath, cur_entry[self.path_cat])
        # class as string
        cur_class_str = cur_entry[self.class_cat]
        # class as int
        cur_class = self.cls_to_idx[cur_class_str]
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
        cur_snd = SL.sndloader(cur_fpath, want_sr=self.srate, frame_offset = cur_offset, max_samp=self.samp_sz, to_mono=True)
        return cur_snd, ret_label


def make_tau_fewshot_tasks(cur_df, classes=[], n_way = 5, k_shot=np.inf, srate = 16000, samp_sz = 118098,  basefolder = UG.DEF_TAUDEVDIR, seed = 3, initial_label_offset = 30, one_hot = True, to_label_tx = True):
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
        cur_ds = TAU(cur_df, classes=cur_classes, k_shot=k_shot, srate=srate, samp_sz=samp_sz, basefolder = basefolder, seed= seed, label_offset = cur_label_offset, one_hot = one_hot, to_label_tx = to_label_tx)
        curtup = (num_classes_to_add, cur_ds)
        ret.append(curtup)
        num_classes_allocated += num_classes_to_add
    return ret 
