import torch
from torch import nn
from collections import OrderedDict
from . import arch_util as AU
from .sampcnn_basic import SampCNNBasic
from .sampcnn_simple import SampCNNSimple
from .sampcnn_resn import SampCNNResN
from .sampcnn_strided import SampCNNStrided
from .sampcnn_se import SampCNNSE

#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479

# if 4s, need 11 blocks for  177147 input size (4.016938775510204s at 44.1 kHz, 8.033877551020408s at 22.05 kHz)
# if 5s, need 11 blocks at 22.05 kHz else 12 for 531441 input size (12.050816326530612 at 44.1 kHz, 24.101632653061223s at 22.05 kHz)

#(num,ksize,stride) = (10,3,3),(2,2,2) gives 236196 which is 15696 extra samples 
# (includes starting strided conv and following regular conv with strided maxpool)
# but doesn't include 1 channel conv

class SampCNNModel(nn.Module):
    def __init__(self, in_ch=1, strided_list=[], basic_list=[], res1_list=[], res2_list=[], se_list=[], rese1_list=[], rese2_list=[], simple_list=[], res1_dropout=0.2, res2_dropout=0.2, rese1_dropout=0.2, rese2_dropout=0.2,simple_dropout=0.5, se_fc_alpha=2.**(-3), rese1_fc_alpha=2.**(-3), rese2_fc_alpha=2.**(-3), use_classifier=True,fc_dim=-1, num_classes=10, sr=44100):
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
        self.use_classifier = use_classifier
        self.num_classes = num_classes
        self.sr = sr

        cur_blklist = []
        # (num, ksize, stride)
        cur_blks = []
        prev_ch = in_ch

        #(1) multiplies channels by 2 if 3rd block after strided or if last "config block"
        # also omits stride 1 conv as found in (3)
        num_strided = 0
        for (num,ks,ch,s) in strided_list:
            for _ in range(num):
                cstr = f"strided{num_strided}" 
                cmodel = SampCNNStrided(conv_in=prev_ch,conv_out=ch,conv_ks=ks,conv_stride=s)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_strided += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))

        num_basic = 0
        for (num,ks,ch,s) in basic_list:
            for _ in range(num):
                cstr = f"basic{num_basic}" 
                cmodel = SampCNNBasic(conv_in = prev_ch, conv_out=ch, conv_ks = ks, mp_ks=ks, mp_stride=s)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_basic += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))

        num_res1 = 0
        for (num,ks,ch,s) in res1_list:
            for _ in range(num):
                cstr = f"resone{num_res1}" 
                cmodel = SampCNNResN(n=1, conv_in = prev_ch, conv_out = ch, conv_ks = ks, dropout=res1_dropout, mp_ks=ks, mp_stride=s, use_se=False)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_res1 += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))


        num_res2 = 0
        for (num,ks,ch,s) in res2_list:
            for _ in range(num):
                cstr = f"restwo{num_res2}" 
                cmodel = SampCNNResN(n=2, conv_in = prev_ch, conv_out = ch, conv_ks = ks, dropout=res1_dropout, mp_ks=ks, mp_stride=s, use_se=False)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_res2 += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))


        num_se = 0
        for (num,ks,ch,s) in se_list:
            for _ in range(num):
                cstr = f"se{num_se}" 
                cmodel = SampCNNSE(conv_in = prev_ch, conv_out = ch, conv_ks = ks, mp_ks=ks, mp_stride=s,fc_alpha=se_fc_alpha)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_se += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))

        num_rese1 = 0
        for (num,ks,ch,s) in rese1_list:
            for _ in range(num):
                cstr = f"reseone{num_rese1}" 
                cmodel = SampCNNResN(n=1,conv_in = prev_ch, conv_out = ch, conv_ks = ks, dropout=rese1_dropout, mp_ks=ks, mp_stride=s,fc_alpha=rese1_fc_alpha, use_se=True)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_rese1 += 1
                cur_blks.append(ctup)
            cur_blklist.append((n,ks,s))


        num_rese2 = 0
        for (num,ks,ch,s) in rese2_list:
            for _ in range(n):
                cstr = f"resetwo{num_rese2}" 
                cmodel = SampCNNReSEN(n=2,conv_in = prev_ch, conv_out = ch, conv_ks = ks, dropout=rese2_dropout, mp_ks=ks, mp_stride=s,fc_alpha=rese2_fc_alpha, use_se=True)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_rese2 += 1
                cur_blks.append(ctup)
            cur_blklist.append((num,ks,s))


        num_simple = 0
        for (num,ks,ch,s) in simple_list:
            for _ in range(num):
                cstr = f"simple{num_simple}" 
                cmodel = SampCNNSimple(conv_in = prev_ch, conv_out =ch, conv_ks=1, dropout=simple_dropout)
                ctup = (cstr, cmodel)
                prev_ch = ch
                num_simple += 1
                cur_blks.append(ctup)
            #cur_blklist.append((n,ks,s))

        self.embedder = nn.Sequential(OrderedDict(cur_blks))
        self.out_ch = prev_ch
        self.in_samples = AU.insize_by_blocks2(cur_blklist)
        self.in_sec = AU.samp_to_sec(self.in_samples,sr=sr)


        # output of embedder should be (n, prev_ch, 1)
        # middle dim according to (1) is same as num channels
        self.classifier = nn.Sequential()
        if use_classifier == True:
            self.fc_dim = fc_dim if fc_dim > 0 else prev_ch
            self.classifier.append(nn.Flatten(start_dim=-2))
            self.classifier.append(nn.Linear(prev_ch, self.fc_dim))
            self.classifier.append(nn.ReLU())
            self.classifier.append(nn.Linear(fc_dim, num_classes))


    def forward(self, cur_ipt):
        emb_out = self.embedder(cur_ipt)
        if self.use_classifier == True:
            net_out = self.classifier(emb_out)
            return net_out
        else:
            return emb_out

