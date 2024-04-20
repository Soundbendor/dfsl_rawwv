import os,sys
import pandas as pd
from ds.esc50 import ESC50, make_esc50_fewshot_tasks 
from ds.tinysol import TinySOL, make_tinysol_fewshot_tasks
from ds.audioset import AudioSet, make_audioset_fewshot_tasks

esc50path = os.path.join(os.sep, 'home', 'dxk', 'ds', 'ESC-50-master')
tinysol_dir = os.path.join(os.sep, 'home', 'dxk', 'ds', 'TinySOL')
audioset_dir = os.path.join(os.sep, 'media', 'dxk', 'TOSHIBA EXT', 'ds', 'audioset', 'ksrc')
audioset_dir2 = os.path.join(os.sep, 'media', 'dxk', 'TOSHIBA EXT', 'ds', 'audioset', 'ksrc2')
tau_2019dir = os.path.join(os.sep, 'media', 'dxk', 'TOSHIBA EXT', 'ds', 'tau_urban_acoustic_scenes_2019')
tau_devdir = os.path.join(tau_2019dir, 'TAU-urban-acoustic-scenes-2019-development')
tau_evaldir = os.path.join(tau_2019dir, 'TAU-urban-acoustic-scenes-2019-openset-evaluation')
tau_leaddir = os.path.join(tau_2019dir, 'TAU-urban-acoustic-scenes-2019-openset-leaderboard')


esc50_df = pd.read_csv(os.path.join(esc50path, "meta", "esc50.csv"))
tinysol_df = pd.read_csv(os.path.join(tinysol_dir, "TinySOL_metadata.csv"))
as_train_df = pd.read_csv(os.path.join(audioset_dir, 'train.csv'))
as_valid_df = pd.read_csv(os.path.join(audioset_dir, 'valid.csv'))
as_cli_df = pd.read_csv(os.path.join(audioset_dir2, 'class_labels_indices.csv'))
tau_devdf = pd.read_csv(os.path.join(tau_devdir, "meta.csv"),sep='\t')
tau_dev_f1evaldf = pd.read_csv(os.path.join(tau_devdir, 'evaluation_setup', "fold1_evaluate.csv"),sep='\t')
tau_dev_f1testdf = pd.read_csv(os.path.join(tau_devdir, 'evaluation_setup', "fold1_test.csv"),sep='\t')
tau_dev_f1traindf = pd.read_csv(os.path.join(tau_devdir, 'evaluation_setup', "fold1_train.csv"),sep='\t')
tau_evaldf = pd.read_csv(os.path.join(tau_evaldir, 'evaluation_setup', "test.csv"),sep='\t')
tau_leaddf = pd.read_csv(os.path.join(tau_leaddir, 'evaluation_setup', "test.csv"),sep='\t')
sr = 16000

esc50_folds = [1,2]
esc50_classes = list(range(14))
tinysol_classes = list(range(12))
as_classes = list(range(527))
tinysol_folds = [1,2]
cur_seed = 3
max_samp = 236196
use_one_hot = True
k_shot = 5
esc50 = ESC50(esc50_df, folds=esc50_folds, classes=esc50_classes, k_shot=24, srate=sr, samp_sz=max_samp, basefolder = esc50path, seed = cur_seed, label_offset = 0, one_hot = use_one_hot, to_label_tx = True)
tinysol = TinySOL(tinysol_df, classes=tinysol_classes, folds = tinysol_folds, k_shot=5, srate=sr, samp_sz=max_samp, basefolder = tinysol_dir, to_label_tx = True, label_offset = 0, one_hot = use_one_hot, seed = cur_seed)
#as_train = AudioSet(as_train_df, dset_type="train", classes=as_classes, k_shot=k_shot, srate=sr, samp_sz=max_samp, basefolder = audioset_dir, to_label_tx = True, label_offset = 0, one_hot = True, seed = 3)
#as_valid = AudioSet(as_valid_df, dset_type="valid", classes=as_classes, k_shot=k_shot, srate=sr, samp_sz=max_samp, basefolder = audioset_dir, to_label_tx = True, label_offset = 0, one_hot = True, seed = 3)
