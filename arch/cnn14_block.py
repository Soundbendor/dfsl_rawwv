import torch
from torch import nn

#REFERENCES:

# (1) Kong, Qiuqiang (2021). audioset_tagging_cnn [Github Repository]. Github. https://github.com/qiuqiangkong/audioset_tagging_cnn
# (2) Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., and Plumbley, M. D. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. IEEE/ACM Transiations on Audio, Speech, and Language Processing, Vol. 2. doi: 10.1109/TASLP.2020.3030497


# (1) does not use a bias term in their CNN
class CNN14Block(nn.Module):
    def __init__(self, conv_in = 128, conv_out = 256, conv_ks = 3, conv_stride = 1, conv_pad = 1, dropout=0.2, ap_ks = 2, omit_last_relu = False, use_prelu = False, use_bias = False):
        super().__init__()
        self.omit_last_relu = omit_last_relu
        self.use_bias = use_bias
        # (1) uses a ks-1 conv1d with batch norm to expand channels

        
        # from this point on, conv_in == conv_out (since layers_exp fixes it)
        self.layers = nn.Sequential()
        self.num_groups = 2
        last_nc = conv_in
        for i in range(self.num_groups):
            omit_relu = i == (self.num_groups) and omit_last_relu == True
            self.layers.append(nn.Conv2d(last_nc, conv_out, (conv_ks,conv_ks), (conv_stride, conv_stride), (conv_pad, conv_pad), bias=use_bias))
            self.layers.append(nn.BatchNorm2d(conv_out))
            if omit_relu == False:
                if use_prelu == True:
                    self.layers.append(nn.PReLU())
                else:
                    self.layers.append(nn.ReLU())
            last_nc = conv_out

        self.layers.append(nn.AvgPool2d((ap_ks,ap_ks)))
        if dropout > 0:
            self.layers.append(nn.Dropout(p=dropout))

    def forward(self, cur_ipt):
        net_out = self.layers(cur_ipt)
        return net_out

