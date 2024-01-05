import torch
from torch import nn

#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479

class SampCNNSE(nn.Module):
    def __init__(self, conv_in = 1, conv_out = 1, conv_ks = 3, mp_ks=3, mp_stride=3,md_pad=0, md_dil=1,fc_alpha=2**4):
        fc_indim = int(conv_out * fc_alpha)
        self.layers = nn.Sequential(
                nn.Conv1d(conv_in, conv_out. conv_ks, stride=1, padding='same',dilation=1),
                nn.BatchNorm1d(conv_out,eps=1e-5,momentum=0.1),
                nn.ReLU(),
                nn.MaxPool1d(mp_ks,mp_stride,mp_pad,mp_dil)
                )
        self.layers2 = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.flatten(start_dim=-2),
                nn.Linear(conv_out, fc_indim),
                nn.ReLU(),
                nn.Linear(fc_indim, conv_out),
                nn.Sigmoid()
                )
        

    def forward(self, cur_ipt):
        first_out = self.layers(cur_ipt)
        second_out = self.layers2(first_out)
        net_out = torch.mul(first_out,second_out.unsqueeze(-1))
        return net_out

