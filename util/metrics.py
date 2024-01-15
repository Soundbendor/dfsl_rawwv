from torch.nn.functional import one_hot
import torch
from torcheval import metrics as TM

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


