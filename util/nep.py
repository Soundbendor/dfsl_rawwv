from util.types import BatchType,TrainPhase
"""
def nep_batch_parser_old(cur_nep, cur_dict,batch_type=BatchType.train, train_phase = TrainPhase.base_init):
    base_str = f"{train_phase.name}/{batch_type.name}"
    cur_nep[f"{base_str}/mean_avg_prec"].append(cur_dict["epoch_avg_ap"])
    cur_nep[f"{base_str}/avg_loss"].append(cur_dict["epoch_avg_loss"])
    cur_nep[f"{base_str}/avg_time"].append(cur_dict["epoch_avg_time"])
    cur_nep[f"{base_str}/avg_top1acc"].append(cur_dict["epoch_avg_acc1"])
"""

def nep_batch_parser(cur_nep, cur_dict, batch_type=BatchType.train, train_phase = TrainPhase.base_init):
    base_str = f"{train_phase.name}/{batch_type.name}"
    for k,v in cur_dict.items():
        cur_nep[f"{base_str}/{k}"].append(v)

def nep_confmat_upload(cur_nep, fpath, batch_type=BatchType.train, train_phase = TrainPhase.base_init):
    base_str = f"{train_phase.name}/{batch_type.name}"
    cur_nep[f"{base_str}/graphs"].upload(fpath)
