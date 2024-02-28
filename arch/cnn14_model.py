import torch
import torchaudio.transforms as TATX
import numpy as np
from torch import nn
from collections import OrderedDict
from . import arch_util as AU
from .cnn14_block import CNN14Block
from .weight_gen_cls import WeightGenCls 
import os,sys
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
from util.types import BatchType,TrainPhase

#REFERENCES:

# (1) Kong, Qiuqiang (2021). audioset_tagging_cnn [Github Repository]. Github. https://github.com/qiuqiangkong/audioset_tagging_cnn
# (2) Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., and Plumbley, M. D. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. IEEE/ACM Transiations on Audio, Speech, and Language Processing, Vol. 2. doi: 10.1109/TASLP.2020.3030497

class CNN14Model(nn.Module):
    def __init__(self, in_ch=1, num_classes_base=10, num_classes_novel = 0, sr=44100, seed=3, omit_last_relu = True, train_phase = TrainPhase.base_init, use_prelu = True, use_bias = False, cls_fn = 'cos_sim'):
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
        num_classes_base: number of classes to classify
        """
        super().__init__()
        self.in_ch = in_ch
        self.num_classes_base = num_classes_base
        self.num_classes_novel = num_classes_novel
        self.sr = sr
        self.train_phase = train_phase
        n_fft = int( AU.ms_to_samp(64,sr=sr))
        win_length = int( AU.ms_to_samp(25, sr=sr))
        hop_size = int( AU.ms_to_samp(10, sr=sr))
        #window = torch.hann_window(window_length=win_length)
        #n_fft = 1024 # 64ms at 16 khz
        #win_length = 400 # 25ms at 16 khz
        #hop_size = 160 # 10ms at 16 khz
        bins = 64
        self.melspect = TATX.MelSpectrogram(sample_rate = sr, n_fft= n_fft,win_length= win_length, hop_length = hop_size, n_mels = bins, window_fn=torch.hann_window)
        ch_pool_tup = [(64,2),(128,2),(256,2),(512,2),(1024,2),(2048,1)]
        # (num, ksize, stride)
        cur_blks = []
        prev_ch = 1
        for i,(ch,p_ks) in enumerate(ch_pool_tup):
            omit_relu = (i == len(ch_pool_tup) - 1) and omit_last_relu == True
            cur_blk = CNN14Block(conv_in = prev_ch, conv_out = ch, conv_ks = 3, conv_stride = 1, conv_pad = 1, dropout=0.2, ap_ks = p_ks, omit_last_relu = omit_relu, use_prelu = use_prelu, use_bias = use_bias)
            blkstr = f"conv{i+1}"
            ctup = (blkstr, cur_blk)
            cur_blks.append(ctup)
            prev_ch = ch
        self.embedder = nn.Sequential(OrderedDict(cur_blks))
        self.out_ch = prev_ch
        #self.in_samples = AU.insize_by_blocks2(cur_blklist)
        #self.in_sec = AU.samp_to_sec(self.in_samples,sr=sr)
        #self.flatten = torch.nn.Flatten(start_dim=-2)
        
        # output of embedder should be (n, prev_ch, 1)
        # middle dim according to (1) is same as num channels
        
        self.classifier = WeightGenCls(num_classes_base = num_classes_base, num_classes_novel = num_classes_novel, dim=ch, seed=seed, train_phase=train_phase, cls_fn=cls_fn)
 
    def freeze_embedder(self,to_freeze):
        self.embedder.requires_grad_(to_freeze==False)

    def freeze_classifier(self,to_freeze):
        self.classifier.freeze_classifier(to_reeze)

    """
    def init_zarr(self, k_novel, k_samp, k_dim, device='cpu'):
        self.zarr = torch.zeros((k_novel, k_samp, k_dim), requires_grad=False).to(device)
        self.zclass = np.zeros(k_novel, dtype=int)
        self.zidx = 0
    """
    def set_train_phase(self, cur_tp):
        self.train_phase = cur_tp
        if cur_tp in [TrainPhase.base_weightgen, TrainPhase.novel_valid, TrainPhase.novel_test, TrainPhase.base_test]:
            self.freeze_embedder(True)
            #self.freeze_classifier()
        else:
            self.freeze_embedder(False)
            #self.unfreeze_classifier()
            #self.zarr = None
            #self.zavg = None
            #self.zclass = None
        self.classifier.set_train_phase(cur_tp)

    def set_exclude_idxs(self, exc_idxs, device='cpu'):
        self.classifier.set_exclude_idxs(exc_idxs, device=device)

    def clear_exclude_idxs(self):
        self.classifier.clear_exclude_idxs()

    def reset_copies(self, copy_back=True, device='cpu'):
        self.classifier.reset_copies(copy_back = copy_back, device=device)

    def weightgen_train_enable(self, to_enable):
        self.classifier.weightgen_train_enable(to_enable)
    """
    def set_zarr(self, k_novel_samp, k_novel_idx):
        k_novel_ft = self.flatten(self.embedder(k_novel_samp))
        self.zarr[self.zidx] = k_novel_ft
        self.zclass[self.zidx] = k_novel_idx
        self.zidx += 1
    """
    """
    def calc_pseudonovel_vecs(self):
        self.zavg = torch.mean(self.zarr, dim=1)
        self.zavg.requires_grad_(False)
        self.watt = torch.zeros_like(self.zavg)
        for i in range(self.zavg.shape[0]):
            self.watt[i] = self.classifier.calc_w_att(self.zarr[i])
        self.classifier.calc_pseudonovel_vecs(self.zarr,self.zavg, self.zclass, self.watt)
    """
    def set_pseudonovel_vec(self, k_novel_idx, k_novel_ex):
        # k_novel_ex should be of size (k_novel, input_dim)
        #print(k_novel_ex.type())
        with (torch.no_grad() if self.train_phase != TrainPhase.base_weightgen else contextlib.nullcontext()):
            k_novel_ft = self.flatten(self.embedder(k_novel_ex))
            self.classifier.set_pseudonovel_vec(k_novel_idx, k_novel_ft)
    """
    def set_pseudonovel_vecs(self, k_novel_idxs, k_novel_sz, k_novel_exs):
        # k_novel_idxs should be of size k_novel
        # k_novel_ex should be of size (all_sizes, input_dim)
        k_novel_fts = self.flatten(self.embedder(k_novel_exs))
        self.classifier.set_pseudonovel_vecs(k_novel_idxs,k_novel_fts)
    """


    def renum_novel_classes(self, num_novel, device='cpu'):
       self.num_classes_novel = self.classifier.renum_novel_classes(num_novel,device=device)

    def forward(self, cur_ipt):
        #torch.Size([5, 1, 64, 1108]) out of spectrogram
        #torch.Size([5, 2048, 2, 34]) out of embeddr

        txed = self.melspect(cur_ipt)
        #print(txed.shape)
        emb_out = self.embedder(txed)
        #print(emb_out.shape)
        
        # borrowing format of (1), this is the global pooling mentioned in (2)
        cmean_w = torch.mean(emb_out, dim=3)
        cmax_h = torch.max(cmean_w, dim=2)[0] # returns tuple of values and indices
        cmean_h = torch.mean(cmean_w, dim=2)
        cm_out = cmax_h + cmean_h 

        #flat_out = self.flatten(cm_out)
        #print(flat_out.shape)
        net_out = self.classifier(cm_out)
        #print(net_out.shape)
        return net_out
        #print(emb_out.shape)

