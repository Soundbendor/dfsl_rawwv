import torch
from torch import nn
from torch.nn.parameter import Parameter

# References:
# (1) Wang, Y., Bryan, N. J., Cartwright, M., Bello, J. P., and Salamon, J. (2021). Few-Shot Continual Learning for Audio Classification. ICASSP 2021 - 2021 IEEE International Conference on Acoustic, Speech and Signal Processing, 321-325. https://doi.org/10.1109/ICASSP39728.2021.9413584.
# (2) Gidaris, S. and Komodakis, N. (2018). Dynamic Few-Shot Visual Learning Without Forgetting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018, (pp. 4367-4375). https://openaccess.thecvf.com/content_cvpr_2018/html/Gidaris_Dynamic_Few-Shot_Visual_CVPR_2018_paper.html.
# (3) Gidaris, S. (2019). FewShotWithoutForgetting [Github Repository]. Github. https://github.com/gidariss/FewShotWithoutForgetting/ 
class WeightGenCls(nn.Module):
    def __init__(self, num_classes = 50, dim=512, seed=3, base_indices = [], training_base=True):
        super().__init__()
        torch.manual_seed(seed)

        
        self.num_classes = num_classes
        self.dim = dim
        self.training_base = training_base
        sdev = torch.sqrt(torch.tensor(2.0/dim)) # from (3)
        cls_vec = torch.randn(num_classes,dim) * sdev # from(3)
        #cls_vec = torch.zeros(num_classes,dim)
        self.cls_vec = Parameter(nn.init.xavier_normal_(cls_vec)) # idea from (3)
        #from (2)
        tau = torch.tensor(10.)
        self.tau = Parameter(tau)

        self.phi_avg = Parameter(torch.randn(dim)*sdev) # idea from (3)ish
        self.phi_att = Parameter(torch.randn(dim)*sdev) # idea from (3)ish
        self.phi_q = Parameter(nn.init.xavier_normal_(torch.zeros(dim,dim))) #idea from (3)ish

        self.include_idxs = []
        self.exclude_idxs = []
    def set_base_class_idxs(self, idxs):
        self.base_classes = idxs

    def forward(self, ipt):
        # input should be (bs, num_channels, length=1)
        # return classifier cosine similarity scores
        
        if self.training_base == True:
        #idea from (3)
            cur_mask = torch.zeros_like(self.cls_vec, requires_grad = False)
            cur_mask[self.base_classes] = 1.
            cos_sim = self.tau * torch.matmul(nn.functional.normalize(ipt,dim=1), nn.functional.normalize(torch.mul(cur_mask, self.cls_vec), dim=1).T)
            return cos_sim
        else:
            cos_sim = self.tau * torch.matmul(nn.functional.normalize(ipt,dim=1), nn.functional.normalize(self.cls_vec, dim=1).T)
        # should be (bs, num_channels) x (num_channels, num_classes) = (bs, num_classes)
