import torch
from torch import nn

#REFERENCES:

# (1) Kong, Qiuqiang (2021). audioset_tagging_cnn [Github Repository]. Github. https://github.com/qiuqiangkong/audioset_tagging_cnn
# (2) Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., and Plumbley, M. D. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. IEEE/ACM Transiations on Audio, Speech, and Language Processing, Vol. 2. doi: 10.1109/TASLP.2020.3030497


# (1) does not use a bias term in their CNN
class CNN14Block(nn.Module):
    def __init__(self, conv_in = 128, conv_out = 256, conv_ks = 3, conv_stride = 1, conv_pad = 1, dropout=0.2, ap_ks = 2, omit_last_relu = False, use_prelu = False, use_bias = False):
        super().__init__()
        
        c1 = nn.Conv2d(conv_in, conv_out, (conv_ks, conv_ks), stride=(conv_stride, conv_stride), padding=(conv_pad, conv_pad), bias=use_bias)
        bn1 = nn.BatchNorm2d(conv_out)
        nl1 = None
        if use_prelu == True:
            nl1 = nn.PReLU()
        else:
            nl1 = nn.ReLU()
        c2 = nn.Conv2d(conv_out, conv_out, (conv_ks, conv_ks), stride=(conv_stride, conv_stride), padding=(conv_pad, conv_pad), bias=use_bias)
        bn2 = nn.BatchNorm2d(conv_out)
        self.layers = nn.Sequential(c1, bn1, nl1, c2, bn2)
        if omit_last_relu == False:
            nl2 = None
            if use_prelu == True:
                nl2 = nn.PReLU()
            else:
                nl2 = nn.ReLU()
            self.layers.append(nl2)
        if ap_ks > 0:
            ap1 = nn.AvgPool2d((ap_ks, ap_ks))
            self.layers.append(ap1)
        if dropout > 0:
            do1 = nn.Dropout(p=dropout)
            self.layers.append(do1)

        #print(self.layers)
    def forward(self, cur_ipt):
        net_out = self.layers(cur_ipt)
        return net_out

