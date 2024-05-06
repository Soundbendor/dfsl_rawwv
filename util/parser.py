import os,sys
import util.globals as UG
from distutils.util import strtobool
import tomllib


settings = {
        "learning_rate": 0.00025,
        "learning_rate_weightgen": 0.0001,
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
        "aug_train": False,
        "aug_wgen": False,
        "aug_novel": False,
        "aug_type": 'random',
        "save_ivl": 0,
        "base_dir": UG.DEF_BASEDIR,
        "novel_dir": UG.DEF_NOVELDIR,
        "save_dir": UG.DEF_SAVEDIR,
        "res_dir": UG.DEF_RESDIR,
        "graph_dir": UG.DEF_GRAPHDIR,
        "load_emb": '',
        "load_cls": '',
        "model_dir": UG.DEF_ROOTDIR,
        "omit_last_relu": True,
        "use_prelu": True,
        "se_prelu": False,
        "to_print": True,
        "to_time": True,
        "to_graph": True,
        "to_res": True,
        "to_nep": True,
        "use_bias": False,
        "emb_expr_num": -1,
        "cls_expr_num": -1,
        "emb_load_num": -1,
        "cls_load_num": -1,
        "emb_idx": 0,
        "cls_idx": 0,
        "train_phase": 'base_init',
        "n_way": 5,
        "use_cuda": True,
        "k_shot": 4,
        "baseset":"esc50",
        "novelset": "esc50",
        }

#parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument("-t", "--toml", type=str, default=os.path.join(UG.DEF_ROOTDIR,"default.toml"), help="toml settings file")

tomlfile = "default.toml"
try:
    if len(sys.argv) > 1:
        tomlfile = sys.argv[1]
        with open(tomlfile, "rb") as f:
            cur_settings = tomllib.load(f)
            #print("got here")
            settings.update(cur_settings)
            print(f"read {tomlfile}")
except:
    print(f"error reading {tomlfile}")
    quit()
