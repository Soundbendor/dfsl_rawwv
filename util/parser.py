import os
import argparse
import util.globals as UG
from distutils.util import strtobool
import tomllib

settings = {
        "learning_rate": 0.00025,
        "model": "samplecnn",
        "cls_fn": 'cos_sim',
        "sample_rate": 16000,
        "batch_size": 5,
        "base_epochs": 1,
        "weightgen_epochs": 10,
        "novel_epochs": 10,
        "multilabel": True,
        "use_class_weights": False,
        "label_smoothing": 0.0,
        "se_fc_alpha": 20.0,
        "rese1_fc_alpha": 20.0,
        "rese2_fc_alpha": 20.0,
        "se_dropout": 0.5,
        "res1_dropout": 0.5,
        "res2_dropout": 0.5,
        "rese1_dropout": 0.5,
        "rese2_dropout": 0.5,
        "simple_dropout": 0.5,
        "dropout": 0.2,
        "dropout_final": 0.5,
        "save_ivl": 0,
        "data_dir": UG.DEF_DATADIR,
        "save_dir": UG.DEF_SAVEDIR,
        "res_dir": UG.DEF_RESDIR,
        "graph_dir": UG.DEF_GRAPHDIR,
        "load_emb": '',
        "load_cls": '',
        "model_dir": UG.DEF_BASEDIR,
        "omit_last_relu": True,
        "use_prelu": True,
        "se_prelu": False,
        "to_print": True,
        "to_time": True,
        "to_graph": True,
        "to_res": True,
        "to_nep": True,
        "use_bias": False,
        "expr_num": -1,
        "emb_load_num": -1,
        "cls_load_num": -1,
        "emb_idx": 0,
        "cls_idx": 0,
        "train_phase": 'base_init',
        "n_way": 5,
        "k_shot": 4,
        "baseset":"esc50",
        "novelset": "esc50",
        }

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--toml", type=str, default=os.path.join(UG.DEF_BASEDIR,"default.toml"), help="toml settings file")
  
    args = parser.parse_args()
    try:
        with open(args.toml, "rb") as f:
            cur_settings = tomllib.load(f)
            settings.update(cur_settings)
            print(f"read {args.toml}")
    except:
        print(f"error reading {args.toml}")
        quit()
    return settings
