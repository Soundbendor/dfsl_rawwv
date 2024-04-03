import torch
from torch import nn
import numpy as np
from torch.nn.parameter import Parameter
import os,sys
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
from util.types import BatchType,TrainPhase

# References:
# (1) Wang, Y., Bryan, N. J., Cartwright, M., Bello, J. P., and Salamon, J. (2021). Few-Shot Continual Learning for Audio Classification. ICASSP 2021 - 2021 IEEE International Conference on Acoustic, Speech and Signal Processing, 321-325. https://doi.org/10.1109/ICASSP39728.2021.9413584.
# (2) Gidaris, S. and Komodakis, N. (2018). Dynamic Few-Shot Visual Learning Without Forgetting. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018, (pp. 4367-4375). https://openaccess.thecvf.com/content_cvpr_2018/html/Gidaris_Dynamic_Few-Shot_Visual_CVPR_2018_paper.html.
# (3) Gidaris, S. (2019). FewShotWithoutForgetting [Github Repository]. Github. https://github.com/gidariss/FewShotWithoutForgetting/ 
class WeightGenCls(nn.Module):
    def __init__(self, num_classes_base = 30, num_classes_novel = 0, dim=512, seed=3, exclude_idxs = [], train_phase=TrainPhase.base_init, cls_fn_type = 'cos_sim'):
        super().__init__()
        torch.manual_seed(seed)

        self.train_phase = train_phase
        self.num_classes_base = num_classes_base
        self.num_classes_novel = num_classes_novel
        self.dim = dim
        #self.base_classes = base_idxs
        self.sdev = torch.sqrt(torch.tensor(2.0/dim)) # from (3)
        self.sdev.requires_grad_(False)

        cls_vec = torch.randn(num_classes_base,dim) * self.sdev # from(3)
        self.cls_vec = Parameter(cls_vec) # idea from (3)

        # separating out base and novel classes like gidaris
        self.cls_vec_novel = torch.zeros(num_classes_novel,dim)
        #self.cls_vec_novel.requires_grad_(False)
        #from (2)
        self.cls_fn_type = 'cos_sim'
        tau = torch.tensor(10.)
        gamma = torch.tensor(10.)
        self.tau = Parameter(tau)
        self.gamma = Parameter(gamma)
        self.phi_avg = Parameter(torch.randn(dim)*self.sdev) # idea from (3)ish
        self.phi_att = Parameter(torch.randn(dim)*self.sdev) # idea from (3)ish
        self.phi_q = Parameter(nn.init.xavier_normal_(torch.zeros(dim,dim))) #idea from (3)ish
        k_b = torch.randn(num_classes_base, dim) * self.sdev # copying init of cls_vec idea from (3)
        self.k_b = Parameter(k_b)
        self.exclude_idxs = np.array(exclude_idxs)
        self.has_exclude_idxs = len(exclude_idxs) > 0

        #self.k_b.requires_grad_(False)
        #self.phi_avg.requires_grad_(False)
        #self.phi_att.requires_grad_(False)
        #self.phi_q.requires_grad_(False)
        self.attn_smax = nn.Softmax(dim=1) #used to take attention over softmax for weight gen
        if self.cls_fn_type == 'cos_sim':
            self.cls_fn = self.cos_sim
        else:
            self.cls_fn = self.rev_euc

    def set_train_phase(self, cur_tph):
        self.train_phase = cur_tph
        
    def clear_exclude_idxs(self):
        self.exclude_idxs = np.array([])
        self.has_exclude_idxs = False

    def cos_sim(self, ipt1, ipt2):
        ret = torch.matmul(nn.functional.normalize(ipt1,dim=1), nn.functional.normalize(ipt2, dim=1).T)
        return ret
   
    def set_exclude_idxs(self, exclude_idxs, device='cpu'):
        self.exclude_idxs = np.array(exclude_idxs)
        self.has_exclude_idxs = self.exclude_idxs.shape[0] > 0


    # change number of classification vector slots
    def renum_novel_classes(self, num_novel_classes,device='cpu'):
        novel_clip_num = max(0, num_novel_classes)
        old_num = self.num_classes_novel
        if novel_clip_num != old_num:
            new_cls_vec_novel = self.cls_vec_novel.clone().detach()
            new_cls_vec_novel.resize_(novel_clip_num, self.dim)
            if old_num > 0:
                new_cls_vec_novel[:old_num] = self.cls_vec_novel[:old_num]
            self.cls_vec_novel = new_cls_vec_novel 
            self.num_classes_novel = novel_clip_num
        return novel_clip_num
        
        

    
    # borrowing indexing idea from (3)
    def get_nonexcluded_idxs(self):
        # get indices that are not excluded
        return np.setdiff1d(np.arange(0, self.num_classes_base), self.exclude_idxs)



    def calc_w_att(self, z_arr):
        #z_arr = (k_shot, dim)
        zq = torch.matmul(z_arr, self.phi_q) # (k_shot, dim) x (dim, dim) = (k_shot, dim)
        # "spiked" cos sim, gives sim scores across rows for each k_shot input
        
        attended = None
        # each row sums over base classes for each k shot input
        if self.has_exclude_idxs == False:
            # doesn't have excluded indices, use simpler indexing scheme
            cur_csim = self.gamma * self.cos_sim(zq, self.k_b) # (k_shot, dim) x (dim, nb) = (k_shot, nb)
            # softmax over the base classes (dim = 1)
            cur_smax = self.attn_smax(cur_csim)
            attended = torch.matmul(cur_smax, nn.functional.normalize(self.cls_vec,dim=1,p=2))
        else:
            nonex_idxs = self.get_nonexcluded_idxs()
            cur_csim = self.gamma * self.cos_sim(zq, self.k_b[nonex_idxs]) # (k_shot, dim) x (dim, nb) = (k_shot, nb)
            cur_smax = self.attn_smax(cur_csim)
            attended = torch.matmul(cur_smax, nn.functional.normalize(self.cls_vec[nonex_idxs], dim=1, p=2))

        kshot_mean = torch.mean(attended, dim=0) # mean over all kshot inputs
        return kshot_mean

    def calc_w_n_plus_1(self, z_arr):
        #print(z_arr.shape, z_avg.shape)
        z_normed = nn.functional.normalize(z_arr,dim=1,p=2)
        w_att = self.calc_w_att(z_normed)
        z_avg = torch.mean(z_normed, dim=0)
        #print(self.phi_avg.requires_grad, self.phi_att.requires_grad, self.phi_q.requires_grad)
        cur_wn = torch.mul(self.phi_avg, z_avg) + torch.mul(self.phi_att, w_att)
        return cur_wn
        
    def set_pseudonovel_vec(self, k_novel_idx, k_novel_ft):
        cur_wn = self.calc_w_n_plus_1(k_novel_ft)
        cur_idx = k_novel_idx - self.num_classes_base
        #print(cur_idx)
        self.cls_vec_novel[k_novel_idx - self.num_classes_base] = cur_wn

   
    def print_cls_vec_norms(self):
        cur_shape = self.cls_vec.shape
        norms = []
        for i in range(cur_shape[0]):
            cur_tup = (i, torch.linalg.vector_norm(self.cls_vec[i,:]).item())
            norms.append(cur_tup)
        print(norms)

    def freeze_classifier(self, to_freeze):
        needs_grad = to_freeze == False
        #print(needs_grad)
        self.cls_vec.requires_grad_(needs_grad)
        self.tau.requires_grad_(needs_grad)

    def weightgen_train_enable(self, to_enable): 
        self.k_b.requires_grad_(to_enable)
        self.phi_avg.requires_grad_(to_enable)
        self.phi_att.requires_grad_(to_enable)
        self.phi_q.requires_grad_(to_enable)
        self.tau.requires_grad_(to_enable)
 
    def calc_pseudonovel_vecs(self, zarrs, zavgs, zclasses, watt):
        for i in range(zarrs.shape[0]):
            self.cls_vec_novel[zclasses[i]- self.num_classes_base]  = self.calc_w_n_plus_1_2(zarrs[i], zavgs[i], watt[i])

    def rev_euc_dist(self, ipt, cls_vec):
        # cdist (b,p,m) x (b,r,m) = (b,p,r)
        # (?, bs=p,nc=r)
        # input is (bs, num_channels) = ie (5, 512)
        # compare to (num_classes_base, num_channels) which is cls_vec
        # output should be (bs, num_classes_base) = ie (5, 30)
        ret = torch.cdist(ipt, cls_vec)
        ret = nn.functional.normalize(ret, dim=1,p=1)
        return (1. - ret)

    def set_pseudonovel_vecs(self, k_novel_idxs, k_novel_sz, k_novel_fts):
        # k_novel_idxs should be of size k_novel
        # k_novel_ft should be of size (all_sizes, dim)
        last_sz = 0
        for i,cur_idx in enumerate(k_novel_idxs):
            cur_sz = k_novel_sz[i]
            cur_wn = self.calc_w_n_plus_1(k_novel_fts[last_sz:cur_sz])
            self.cls_vec_novel[cur_idx - self.num_classes_base] = cur_wn
            last_sz = cur_sz


    def forward(self, ipt):
        # input should be (bs, num_channels)
        # return classifier cosine similarity scores
        ret_base = None
        ret = None

        if self.has_exclude_idxs == False:
            # no excluded indices, just do as normal 
            ret_base = self.cls_fn(ipt, self.cls_vec)
        else:
            bs = ipt.shape[0]
            # has excluded indices, just fill in everything excluded as 0
            ret_base = torch.zeros(bs, self.num_classes_base)
            nonex_idxs = self.get_nonexcluded_idxs()
            nonex_scores = self.tau * self.cls_fn(ipt, self.cls_vec[nonex_idxs])
            ret_base[:,nonex_idxs] = nonex_scores
        if self.num_classes_novel > 0:
            ret_novel = self.tau * self.cls_fn(ipt, self.cls_vec_novel)
            ret = torch.hstack((ret_base, ret_novel))
            #print("jackpot")
        else:
            ret = ret_base
        #print(ret.requires_grad)
        return ret
        # should be (bs, num_channels) x (num_channels, num_classes_base) = (bs, num_classes_base)
