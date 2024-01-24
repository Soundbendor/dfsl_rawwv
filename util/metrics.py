from torch.nn.functional import one_hot
import torch
from torcheval import metrics as TM


#macro_recall right now is buggy https://github.com/pytorch/torcheval/issues/189

# track these classes per batch
not_printable = ["confmat"]
#metric_keys = ["micro_acc1", "macro_acc1","micro_acc3", "macro_acc3", "confmat", "avg_prec", "micro_f1", "macro_f1", "micro_prec", "macro_prec", "micro_recall", "macro_recall", "auroc"]
metric_keys = ["micro_acc1", "macro_acc1","micro_acc3", "macro_acc3", "confmat", "avg_prec", "micro_f1", "macro_f1", "micro_prec", "macro_prec", "micro_recall", "auroc"]
csvable = list(set(metric_keys).difference(set(not_printable)))

def metric_array_updater(update_valdict, cur_arr_dict=None):
    ret_dict = None
    if cur_arr_dict == None:
        ret_dict = {k:[] for k in update_valdict.items()}
    for k,v in update_valdict.items():
        ret_dict[k].append(v)
    return ret_dict

def metric_updater(cur_dict, ipt_logit, ground_truth):
    lo2 = ipt_logit.detach().clone().to(ground_truth.device)
    for k,v in cur_dict.items():
        v.update(lo2, ground_truth)

def metric_printer(cur_valdict):
    printstr = ""
    for k,v in cur_valdict.items():
        if k not in not_printable:
            cur_str = f"{k}: {v}\n"
            printstr += cur_str
    print(printstr)

def metric_compute(cur_dict):
    mdict={k:v.compute() for k,v in cur_dict.items()}
    """
    mdict = {}
    for k,v in cur_dict.items():
        print(k)
        mdict[k] = v.compute()
    """
    return mdict

def metric_creator(num_classes=50):
    mdict = {}
    mdict["micro_acc1"] = TM.MulticlassAccuracy(average='micro', num_classes=num_classes, k=1)
    mdict["macro_acc1"] = TM.MulticlassAccuracy(average='macro', num_classes=num_classes, k=1)
    mdict["micro_acc3"] = TM.MulticlassAccuracy(average='micro', num_classes=num_classes, k=3)
    mdict["macro_acc3"] = TM.MulticlassAccuracy(average='macro', num_classes=num_classes, k=3)
    mdict["confmat"] = TM.MulticlassConfusionMatrix(num_classes=num_classes)
    mdict["avg_prec"] = TM.MulticlassAUPRC(num_classes=num_classes)
    mdict["micro_f1"] = TM.MulticlassF1Score(num_classes=num_classes, average="micro")
    mdict["macro_f1"] = TM.MulticlassF1Score(num_classes=num_classes, average="macro")
    mdict["micro_prec"] = TM.MulticlassPrecision(num_classes=num_classes, average="micro")
    mdict["macro_prec"] = TM.MulticlassPrecision(num_classes=num_classes, average="macro")
    mdict["micro_recall"] = TM.MulticlassRecall(num_classes=num_classes, average="micro")
    #mdict["macro_recall"] = TM.MulticlassRecall(num_classes=num_classes, average="macro")
    mdict["auroc"] = TM.MulticlassAUROC(num_classes=num_classes)
    return mdict

def top1_acc(logits, ground_truth):
    lo2 = logits.detach().clone().to(ground_truth.device)
    #got_right = torch.isclose(lo2.argmax(dim=1), ground_truth)
    #acc = torch.mean(torch.where(got_right, 1., 0.)).item()
    acc= TM.functional.multiclass_accuracy(lo2, ground_truth)
    return acc


def top1_acc2(logits, one_hot_labels):
    lo2 = logits.detach().clone().to(one_hot_labels.device)
    lonehot =  one_hot(lo2.argmax(dim=1), num_classes = lo2.shape[1])
    got_right = torch.all(torch.eq(lonehot, one_hot_labels))
    acc = torch.mean(torch.where(got_right, 1., 0.)).item()
    return acc

def avg_prec(logits, ground_truth, num_classes = 50):
    lo2 = logits.detach().clone().to(ground_truth.device)
    ap = TM.functional.multiclass_precision(lo2,ground_truth,num_classes=num_classes,average='macro')
    return ap


