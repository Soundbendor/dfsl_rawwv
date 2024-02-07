from torch.nn.functional import one_hot
import torch
from torcheval import metrics as TM
import torchmetrics.classification as TMC


#macro_recall right now is buggy https://github.com/pytorch/torcheval/issues/189

# track these classes per batch
not_printable = ["confmat", "multilabel"]
ignorekeys = ["multilabel"]
#metric_keys = ["micro_acc1", "macro_acc1","micro_acc3", "macro_acc3", "confmat", "avg_prec", "micro_f1", "macro_f1", "micro_prec", "macro_prec", "micro_recall", "macro_recall", "auroc"]
metric_keys_mc = ["acc1_micro", "acc1_macro","acc3_micro", "acc3_macro", "confmat", "avgprec", "f1_micro", "f1_macro", "prec_micro", "prec_macro", "recall_micro", "auroc"]
csvable_mc = list(set(metric_keys_mc).difference(set(not_printable)))
metric_keys_ml = ["hamming_macro","hamming_micro", "exact_match", "f1_macro", "f1_micro", "acc_macro", "acc_micro", "prec_macro", "prec_micro", "avgprec_macro", "avgprec_micro", "auroc_macro", "auroc_micro", "confmat"]
csvable_ml = list(set(metric_keys_ml).difference(set(not_printable)))
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
        if k not in ignorekeys:
            v.update(lo2, ground_truth)

def metric_printer(cur_valdict):
    printstr = ""
    for k,v in cur_valdict.items():
        if k not in not_printable:
            cur_str = f"{k}: {v}\n"
            printstr += cur_str
    print(printstr)

def metric_compute(cur_dict):
    mdict={k:v.compute() for k,v in cur_dict.items() if k not in ignorekeys}
    mdict["multilabel"] = cur_dict["multilabel"]
    """
    mdict = {}
    for k,v in cur_dict.items():
        print(k)
        mdict[k] = v.compute()
    """
    return mdict

def metric_creator(num_classes=50, multilabel = False, threshold = 0.5):
    mdict = {"multilabel": multilabel}
    if multilabel == False:
        mdict["acc1_micro"] = TM.MulticlassAccuracy(average='micro', num_classes=num_classes, k=1)
        mdict["acc1_macro"] = TM.MulticlassAccuracy(average='macro', num_classes=num_classes, k=1)
        mdict["acc3_micro"] = TM.MulticlassAccuracy(average='micro', num_classes=num_classes, k=3)
        mdict["acc3_macro"] = TM.MulticlassAccuracy(average='macro', num_classes=num_classes, k=3)
        mdict["confmat"] = TM.MulticlassConfusionMatrix(num_classes=num_classes)
        mdict["avgprec"] = TM.MulticlassAUPRC(num_classes=num_classes)
        mdict["f1_micro"] = TM.MulticlassF1Score(num_classes=num_classes, average="micro")
        mdict["f1_macro"] = TM.MulticlassF1Score(num_classes=num_classes, average="macro")
        mdict["prec_micro"] = TM.MulticlassPrecision(num_classes=num_classes, average="micro")
        mdict["prec_macro"] = TM.MulticlassPrecision(num_classes=num_classes, average="macro")
        mdict["recall_micro"] = TM.MulticlassRecall(num_classes=num_classes, average="micro")
        #mdict["macro_recall"] = TM.MulticlassRecall(num_classes=num_classes, average="macro")
        mdict["auroc"] = TM.MulticlassAUROC(num_classes=num_classes)
    else:
        mdict["hamming_macro"] = TMC.HammingDistance(task="multilabel", threshold=threshold, average="macro", num_labels = num_classes)
        mdict["hamming_micro"] = TMC.HammingDistance(task="multilabel", threshold=threshold, average="micro", num_labels = num_classes)
        mdict["exact_match"] = TMC.ExactMatch(task="multilabel", num_labels= num_classes)
        mdict["f1_macro"] = TMC.F1Score(task="multilabel", threshold=threshold, average="macro", num_labels = num_classes)
        mdict["f1_micro"] = TMC.F1Score(task="multilabel", threshold=threshold, average="micro", num_labels = num_classes)
        mdict["acc_macro"] = TMC.Accuracy(task="multilabel", threshold=threshold, average="macro", num_labels = num_classes)
        mdict["acc_micro"] = TMC.Accuracy(task="multilabel", threshold=threshold, average="micro", num_labels = num_classes)
        mdict["prec_macro"] = TMC.Precision(task="multilabel", threshold=threshold, average="macro", num_labels = num_classes)
        mdict["prec_micro"] = TMC.Precision(task="multilabel", threshold=threshold, average="micro", num_labels = num_classes)
        mdict["avgprec_macro"] = TMC.AveragePrecision(task="multilabel", average="macro", num_labels = num_classes)
        mdict["avgprec_micro"] = TMC.AveragePrecision(task="multilabel", average="micro", num_labels = num_classes)
        mdict["auroc_macro"] = TMC.AUROC(task="multilabel", average="macro", num_labels = num_classes)
        mdict["auroc_micro"] = TMC.AUROC(task="multilabel", average="micro", num_labels = num_classes)
        mdict["confmat"] = TMC.ConfusionMatrix(task="multilabel", threshold=threshold, num_labels = num_classes)

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


