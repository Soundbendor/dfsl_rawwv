from torch.nn.functional import one_hot
import torch

def top1_acc(logits, one_hot_labels):
    lo2 = logits.detach().clone().to(one_hot_labels.device)
    lonehot =  one_hot(lo2.argmax(dim=1), num_classes = lo2.shape[1])
    got_right = torch.all(torch.eq(lonehot, one_hot_labels))
    acc = torch.mean(torch.where(got_right, 1., 0.)).item()
    return acc
