import torch
from torch import nn

#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479


class SampCNNSimple(nn.Module):
    def __init__(self, conv_in = 1, conv_out = 1, conv_ks=1, dropout=0.5):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv1d(conv_in, conv_out, conv_ks, stride=1, padding="same",dilation=1),
                nn.ReLU(),
                nn.Dropout(p=dropout)
                )

    
    def forward(self, cur_ipt):
        net_out = self.layers(cur_ipt)
        return net_out

