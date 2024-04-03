import os,sys
import pandas as pd
from ds.esc50 import ESC50, make_esc50_fewshot_tasks 
from ds.tinysol import TinySOL, make_tinysol_fewshot_tasks

esc50path = os.path.join(os.sep, 'home', 'dxk', 'ds', 'ESC-50-master')
tinysol_dir = os.path.join(os.sep, 'home', 'dxk', 'ds', 'TinySOL')

esc50_df = pd.read_csv(os.path.join(esc50path, "meta", "esc50.csv"))
tinysol_df = pd.read_csv(os.path.join(tinysol_dir, "TinySOL_metadata.csv"))

sr = 16000

esc50_folds = [1,2]
esc50_classes = list(range(14))
tinysol_classes = list(range(12))
tinysol_folds = [1,2]
cur_seed = 3
max_samp = 236196
use_one_hot = True
k_shot = 5
esc50 = ESC50(esc50_df, folds=esc50_folds, classes=esc50_classes, k_shot=24, srate=sr, samp_sz=max_samp, basefolder = esc50path, seed = cur_seed, label_offset = 0, one_hot = use_one_hot, to_label_tx = True)
tinysol = TinySOL(tinysol_df, classes=tinysol_classes, folds = tinysol_folds, k_shot=5, srate=sr, samp_sz=max_samp, basefolder = tinysol_dir, to_label_tx = True, label_offset = 0, one_hot = use_one_hot, seed = cur_seed)

