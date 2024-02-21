import os,sys,csv
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
import util.globals as UG
import pandas as pd

def get_labels(cur_df):
    ret = {}
    for lgroup in cur_df["positive_labels"]:
        cur = [x.strip() for x in lgroup.split(",")]
        for x in cur:
            if x in ret.keys():
                ret[x] += 1
            else:
                ret[x] = 1
    ret_ser = pd.Series(data=ret)
    return ret_ser
def get_label_list(cur_df):
    ret = []
    for lgroup in cur_df["positive_labels"]:
        cur = [x.strip() for x in lgroup.split(",")]
        ret += cur
    ret = sorted(list(set(ret)))
    return ret
train_csv = os.path.join(UG.DEF_AUDIOSETDIR, "train.csv")
valid_csv = os.path.join(UG.DEF_AUDIOSETDIR, "valid.csv")
labels_csv = os.path.join(UG.DEF_AUDIOSETDIR, "class_labels_indices.csv")

tdf = pd.read_csv(train_csv)
vdf = pd.read_csv(valid_csv)
ldf = pd.read_csv(labels_csv)

tl = get_labels(tdf)
vl = get_labels(vdf)

tl2 = get_label_list(tdf)
vl2 = get_label_list(vdf)
