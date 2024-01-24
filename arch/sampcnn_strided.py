import torch
from torch import nn

#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479


class SampCNNStrided(nn.Module):
    def __init__(self, conv_in = 1, conv_out = 1, conv_ks = 3, conv_stride = 3, omit_last_relu = False, use_prelu = True):
        super().__init__()
        self.omit_last_relu = omit_last_relu
        self.layers = nn.Sequential(
                nn.Conv1d(conv_in, conv_out, conv_ks, stride=conv_stride, padding=0,dilation=1),
                nn.BatchNorm1d(conv_out,eps=1e-5,momentum=0.1))
        if omit_last_relu == False:
            if use_prelu == True:
                self.layers.append(nn.PReLU())
            else:
                self.layers.append(nn.ReLU())

    def forward(self, cur_ipt):
        net_out = self.layers(cur_ipt)
        return net_out

