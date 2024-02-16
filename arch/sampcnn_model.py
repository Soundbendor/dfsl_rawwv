import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from . import arch_util as AU
from .sampcnn_basic import SampCNNBasic
from .sampcnn_simple import SampCNNSimple
from .sampcnn_resn import SampCNNResN
from .sampcnn_strided import SampCNNStrided
from .sampcnn_se import SampCNNSE
from .weight_gen_cls import WeightGenCls 
import os,sys
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
from util.types import BatchType,TrainPhase

#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479

# if 4s, need 11 blocks for  177147 input size (4.016938775510204s at 44.1 kHz, 8.033877551020408s at 22.05 kHz)
# if 5s, need 11 blocks at 22.05 kHz else 12 for 531441 input size (12.050816326530612 at 44.1 kHz, 24.101632653061223s at 22.05 kHz)

#(num,ksize,stride) = (10,3,3),(2,2,2) gives 236196 which is 15696 extra samples 
# (includes starting strided conv and following regular conv with strided maxpool)
# but doesn't include 1 channel conv

#80000 samples = 5 sec at 16khz
# (8,3,3) (4,2,2) gives 104976 samp
# (10,3,3) (,1,2,2) gives 118098 samp
# (11,3,3) gives 177147 samp


class SampCNNModel(nn.Module):
    def __init__(self, in_ch=1, strided_list=[], basic_list=[], res1_list=[], res2_list=[], se_list=[], rese1_list=[], rese2_list=[], simple_list=[], se_dropout=0.2, res1_dropout=0.2, res2_dropout=0.2, rese1_dropout=0.2, rese2_dropout=0.2,simple_dropout=0.5, se_fc_alpha=2.**(-3), rese1_fc_alpha=2.**(-3), rese2_fc_alpha=2.**(-3), num_classes=10, sr=44100, seed=3, omit_last_relu = True, train_phase = TrainPhase.base_init, use_prelu = True, se_prelu = False, cls_fn = 'cos_sim'):
        """
        EMBEDDER Layers (stored in self.embedder)
        strided_list: tuples of (num, ksize, out_channels, stride)
        basic_list: tuples of (num, ksize, out_channels, stride)
        res1_list: tuples of (num, ksize, out_channels, stride)
        res2_list: tuples of (num, ksize, out_channels, stride)
        se_list: tuples of (num, ksize, out_channels, stride)
        rese1_list: tuples of (num, ksize, out_channels, stride)
        rese2_list: tuples of (num, ksize, out_channels, stride)
        simple_list: tuples of (num, ksize, out_channels, stride) (channels ignored since only one channel)

        Adds in order stride,basic, res1, res2, se, rese1, rese2, simple

        CLASSIFIER Layers (stored in self.classifier):
        use_classifier: to use a classifier or not
        fc_dim: inner dimension of classifier
        num_classes: number of classes to classify
        """
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.sr = sr

        cur_blklist = []
        # (num, ksize, stride)
        cur_blks = []
        prev_ch = in_ch

        #(1) multiplies channels by 2 if 3rd block after strided or if last "config block"
        # also omits stride 1 conv as found in (3)
        num_strided = 0
        rese2_isfinal = len(simple_list) == 0
        rese1_isfinal = len(rese2_list) == 0 and rese2_isfinal
        se_isfinal = len(rese1_list) == 0 and rese1_isfinal
        res2_isfinal = len(se_list) == 0 and se_isfinal
        res1_isfinal = len(res2_list) == 0 and res2_isfinal
        basic_isfinal = len(res1_list) == 0 and res1_isfinal
        strided_isfinal = len(basic_list) == 0 and basic_isfinal
        for (num,ks,ch,s) in strided_list:
            for i in range(num):
                cstr = f"strided{num_strided}" 
                cmodel = SampCNNStrided(conv_in=prev_ch,conv_out=ch,conv_ks=ks,conv_stride=s,omit_last_relu=((i == num-1) and strided_isfinal), use_prelu = use_prelu)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_strided += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))

        num_basic = 0
        for (num,ks,ch,s) in basic_list:
            for i in range(num):
                cstr = f"basic{num_basic}" 
                cmodel = SampCNNBasic(conv_in = prev_ch, conv_out=ch, conv_ks = ks, mp_ks=ks, mp_stride=s,omit_last_relu=((i == num-1) and basic_isfinal and omit_last_relu), use_prelu = use_prelu)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_basic += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))

        num_res1 = 0
        for (num,ks,ch,s) in res1_list:
            for i in range(num):
                cstr = f"resone{num_res1}" 
                cmodel = SampCNNResN(n=1, conv_in = prev_ch, conv_out = ch, conv_ks = ks, dropout=res1_dropout, mp_ks=ks, mp_stride=s, use_se=False, omit_last_relu=((i == num-1) and res1_isfinal and omit_last_relu), use_prelu = use_prelu)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_res1 += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))


        num_res2 = 0
        for (num,ks,ch,s) in res2_list:
            for i in range(num):
                cstr = f"restwo{num_res2}" 
                cmodel = SampCNNResN(n=2, conv_in = prev_ch, conv_out = ch, conv_ks = ks, dropout=res1_dropout, mp_ks=ks, mp_stride=s, use_se=False, omit_last_relu=((i == num-1) and res2_isfinal and omit_last_relu), use_prelu = use_prelu)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_res2 += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))


        num_se = 0
        for (num,ks,ch,s) in se_list:
            for i in range(num):
                cstr = f"se{num_se}" 
                cmodel = SampCNNSE(conv_in = prev_ch, conv_out = ch, conv_ks = ks, mp_ks=ks, mp_stride=s,fc_alpha=se_fc_alpha, dropout = se_dropout, omit_last_relu=((i == num-1) and se_isfinal and omit_last_relu), use_prelu = use_prelu, se_prelu = se_prelu)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_se += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))

        num_rese1 = 0
        for (num,ks,ch,s) in rese1_list:
            for i in range(num):
                cstr = f"reseone{num_rese1}" 
                cmodel = SampCNNResN(n=1,conv_in = prev_ch, conv_out = ch, conv_ks = ks, dropout=rese1_dropout, mp_ks=ks, mp_stride=s,fc_alpha=rese1_fc_alpha, use_se=True, omit_last_relu=((i == num-1) and rese1_isfinal and omit_last_relu), use_prelu = use_prelu, se_prelu = se_prelu)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_rese1 += 1
                cur_blks.append(ctup)
            cur_blklist.append((n,ks,s))


        num_rese2 = 0
        for (num,ks,ch,s) in rese2_list:
            for i in range(num):
                cstr = f"resetwo{num_rese2}" 
                cmodel = SampCNNResN(n=2,conv_in = prev_ch, conv_out = ch, conv_ks = ks, dropout=rese2_dropout, mp_ks=ks, mp_stride=s,fc_alpha=rese2_fc_alpha, use_se=True, omit_last_relu=((i == num-1) and rese2_isfinal and omit_last_relu), use_prelu = use_prelu, se_prelu = se_prelu)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_rese2 += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))


        num_simple = 0
        for (num,ks,ch,s) in simple_list:
            for i in range(num):
                cstr = f"simple{num_simple}" 
                cmodel = SampCNNSimple(conv_in = prev_ch, conv_out =ch, conv_ks=1, dropout=simple_dropout, omit_last_relu=((i == num-1) and omit_last_relu), use_prelu = use_prelu)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_simple += 1
                cur_blks.append(ctup)
            #cur_blklist.append((n,ks,s))

        self.embedder = nn.Sequential(OrderedDict(cur_blks))
        self.out_ch = prev_ch
        self.in_samples = AU.insize_by_blocks2(cur_blklist)
        self.in_sec = AU.samp_to_sec(self.in_samples,sr=sr)
        self.flatten = torch.nn.Flatten(start_dim=-2)
        
        # output of embedder should be (n, prev_ch, 1)
        # middle dim according to (1) is same as num channels
        
        self.classifier = WeightGenCls(num_classes = num_classes, dim=prev_ch, seed=seed, train_phase=TrainPhase.base_init, cls_fn=cls_fn)
        """
        self.classifier = nn.Sequential()
        if use_classifier == True:
            self.fc_dim = fc_dim if fc_dim > 0 else prev_ch
            self.classifier.append(nn.Flatten(start_dim=-2))
            self.classifier.append(nn.Linear(prev_ch, self.fc_dim))
            self.classifier.append(nn.ReLU())
            self.classifier.append(nn.Linear(fc_dim, num_classes))
        """
  
    def freeze_embedder(self):
        self.embedder.requires_grad_(False)

    def unfreeze_embedder(self):
        self.embedder.requires_grad_(True)

    def freeze_classifier(self):
        self.classifier.freeze_classifier()

    def unfreeze_classifier(self):
        self.classifier.unfreeze_classifier()

    def init_zarr(self, k_novel, k_samp, k_dim, device='cpu'):
        self.zarr = torch.zeros((k_novel, k_samp, k_dim), requires_grad=False).to(device)
        self.zclass = np.zeros(k_novel, dtype=int)
        self.zidx = 0

    def set_train_phase(self, cur_tp):
        if cur_tp == TrainPhase.base_weightgen:
            self.freeze_embedder()
            self.freeze_classifier()
        else:
            self.unfreeze_embedder()
            self.unfreeze_classifier()
            self.zarr = None
            self.zavg = None
            self.zclass = None
        self.classifier.set_train_phase(cur_tp)

    def set_exclude_idxs(self, exc_idxs):
        self.classifier.set_exclude_idxs(exc_idxs)

    def clear_exclude_idxs(self):
        self.classifier.clear_exclude_idxs()

    def reset_copies(self):
        self.classifier.reset_copies()

    def set_zarr(self, k_novel_samp, k_novel_idx):
        k_novel_ft = self.flatten(self.embedder(k_novel_samp))
        self.zarr[self.zidx] = k_novel_ft
        self.zclass[self.zidx] = k_novel_idx
        self.zidx += 1

    def calc_pseudonovel_vecs(self):
        self.zavg = torch.mean(self.zarr, dim=1)
        self.zavg.requires_grad_(False)
        self.watt = torch.zeros_like(self.zavg)
        for i in range(self.zavg.shape[0]):
            self.watt[i] = self.classifier.calc_w_att(self.zarr[i])
        self.classifier.calc_pseudonovel_vecs(self.zarr,self.zavg, self.zclass, self.watt)

    def set_pseudonovel_vec(self, k_novel_idx, k_novel_ex):
        # k_novel_ex should be of size (k_novel, input_dim)
        #print(k_novel_ex.type())
        k_novel_ft = self.flatten(self.embedder(k_novel_ex))
        self.classifier.set_pseudonovel_vec(k_novel_idx, k_novel_ft)

    def set_pseudonovel_vecs(self, k_novel_idxs, k_novel_sz, k_novel_exs):
        # k_novel_idxs should be of size k_novel
        # k_novel_ex should be of size (all_sizes, input_dim)
        k_novel_fts = self.flatten(self.embedder(k_novel_exs))
        self.classifier.set_pseudonovel_vecs(k_novel_idxs,k_novel_fts)

    def forward(self, cur_ipt):
        emb_out = self.embedder(cur_ipt)
        #print(emb_out.shape)
        flat_out = self.flatten(emb_out)
        #print(flat_out.shape)
        net_out = self.classifier(flat_out)
        return net_out
        #print(emb_out.shape)

