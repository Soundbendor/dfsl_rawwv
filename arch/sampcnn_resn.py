import torch
from torch import nn

#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479


# defining res-n and resen as in (1) in one class

class SampCNNResN(nn.Module):
    def __init__(self, n=1, conv_in = 1, conv_out = 1, conv_ks = 3, dropout=0.2, mp_ks=3, mp_stride=3,mp_pad=0, mp_dil=1, fc_alpha=2**4, use_se = False, omit_last_relu = False, use_prelu = True, se_prelu = False):
        super().__init__()
        self.n = max(n,1)
        self.use_se = use_se
        self.expand_ch = conv_in != conv_out
        fc_indim = int(conv_out * fc_alpha)
        self.omit_last_relu = omit_last_relu
        # (1) uses a ks-1 conv1d with batch norm to expand channels
        if self.expand_ch == True:
            self.layers_exp = nn.Sequential()
            self.layers_exp.append(nn.Conv1d(conv_in, conv_out, 1, stride=1, padding='same',dilation=1))
            self.layers_exp.append(nn.BatchNorm1d(conv_out,eps=1e-5,momentum=0.1))

        
        # from this point on, conv_in == conv_out (since layers_exp fixes it)
        self.layers = nn.Sequential()
        if self.n >= 2:
            for i in range(self.n-1):
                self.layers.append(nn.Conv1d(conv_out, conv_out, conv_ks, stride=1, padding='same',dilation=1))
                self.layers.append(nn.BatchNorm1d(conv_out,eps=1e-5,momentum=0.1))
                if use_prelu == True:
                    self.layers.append(nn.PReLU())
                else:
                    self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout))

        self.layers.append(nn.Conv1d(conv_out,conv_out,conv_ks,stride=1,padding='same',dilation=1))
        self.layers.append(nn.BatchNorm1d(conv_out,eps=1e-5,momentum=0.1))

        if use_se == True:
            self.layers_se = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(start_dim=-2),
                    nn.Linear(conv_out, fc_indim))
            if se_prelu == True:
                self.layers_se.append(nn.PReLU())
            else:
                self.layers_se.append(nn.ReLU())
            self.layers_se.append(nn.Linear(fc_indim, conv_out))
            self.layers_se.append(nn.Sigmoid())
                    
     
        self.layers2 = nn.Sequential()
        if omit_last_relu == False:
            if use_prelu == True:
                self.layers2.append(nn.PReLU())
            else:
                self.layers2.append(nn.ReLU())
        self.layers2.append(nn.MaxPool1d(mp_ks,mp_stride,mp_pad,mp_dil))

    def forward(self, cur_ipt):
        scl_ipt = None
        if self.expand_ch == True:
            scl_ipt = self.layers_exp(cur_ipt)
        else:
            scl_ipt = cur_ipt
        first_out = self.layers(scl_ipt)
        pre_residual = None
        if self.use_se == True:
            second_out = self.layers_se(first_out)
            pre_residual = torch.mul(first_out,second_out.unsqueeze(-1))
        else:
            pre_residual = first_out
        residual = pre_residual + scl_ipt
        net_out = self.layers2(residual)
        return net_out

