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
    def __init__(self, num_classes_base = 30, num_classes_novel = 0, dim=512, seed=3, exclude_idxs = [], train_phase=TrainPhase.base_init, cls_fn = 'cos_sim'):
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
        #cls_vec = torch.zeros(num_classes_base,dim)
        #self.cls_vec = Parameter(nn.init.xavier_normal_(cls_vec)) # idea from (3)
        self.cls_vec = Parameter(cls_vec) # idea from (3)
        #from (2)
        self.cls_fn = 'cos_sim'
        tau = torch.tensor(10.)
        gamma = torch.tensor(10.)
        self.tau = Parameter(tau)
        self.gamma = Parameter(gamma)
        self.phi_avg = Parameter(torch.randn(dim)*self.sdev) # idea from (3)ish
        self.phi_att = Parameter(torch.randn(dim)*self.sdev) # idea from (3)ish
        self.phi_q = Parameter(nn.init.xavier_normal_(torch.zeros(dim,dim))) #idea from (3)ish
        k_b = torch.randn(num_classes_base, dim) * self.sdev # copying init of cls_vec idea from (3)
        self.k_b = Parameter(k_b)
        self.include_idxs = []
        self.exclude_idxs = []
        self.k_b.requires_grad_(False)
        self.phi_avg.requires_grad_(False)
        self.phi_att.requires_grad_(False)
        self.phi_q.requires_grad_(False)
        self.attn_smax = nn.Softmax(dim=1) #used to take attention over softmax for weight gen

    def set_train_phase(self, cur_tph):
        self.train_phase = cur_tph
        
    def clear_exclude_idxs(self):
        self.exclude_idxs.clear()

    def cos_sim(self, ipt1, ipt2):
        ret = torch.matmul(nn.functional.normalize(ipt1,dim=1), nn.functional.normalize(ipt2, dim=1).T)
        return ret
   
    def append_include_idxs(self, idxs):
        self.include_idxs += idxs
        #sorted(self.include_idxs)

    def set_exclude_idxs(self, exclude_idxs, device='cpu'):
        self.exclude_idxs = exclude_idxs
        try:
            del self.cls_vec_copy
            del self.cls_vec_pseudo
            del self.k_b_copy
        except:
            pass
        self.cls_vec_copy = Parameter(self.cls_vec.clone().detach()).to(device)
        self.cls_vec_copy.requires_grad_(False)
        self.cls_vec_copy[exclude_idxs] = 0.
        self.cls_vec_pseudo = self.cls_vec_copy.clone().detach()
        self.cls_vec_pseudo.requires_grad_(False)
        #self.cls_vec_copy.requires_grad_(True)

        #self.k_b_copy = self.k_b.clone().detach()
        #self.k_b_copy[exclude_idxs] = 0.
        #self.k_b_copy.requires_grad_(True)
        #self.k_b.requires_grad_(False)

    def reset_copies(self, copy_back = True, device='cpu'):
        if copy_back == True:
            if self.cls_vec_copy != None:
                self.cls_vec_copy = Parameter(self.cls_vec_copy.detach()).to(device)
                self.cls_vec = Parameter(self.cls_vec.detach()).to(device)
                self.cls_vec.requires_grad_(False)
                cur_sz = self.cls_vec.shape[0]
                idxs_to_set = np.setdiff1d(np.arange(0,cur_sz,1), self.exclude_idxs)
                self.cls_vec[idxs_to_set,:] = self.cls_vec_copy[idxs_to_set,:]
                self.cls_vec.requires_grad_(True)
        self.exclude_idxs = []
        try:
            del self.cls_vec_copy
            del self.cls_vec_pseudo
            #del self.k_b_copy

            self.cls_vec_copy = None
            self.cls_vec_pseudo = None
            #self.k_b_copy = None

        except:
            pass
        #self.k_b_copy.requires_grad_(False)
        #self.k_b_copy[self.exclude_idxs] = self.k_b[self.exclude_idxs]
        #self.k_b = nn.Parameter(self.k_b_copy.clone().detach())
        #self.k_b.requires_grad_(True)


    # change number of classification vector slots
    def renum_novel_classes(self, num_novel_classes,device='cpu'):
        grad_was_true = self.cls_vec.requires_grad
        if grad_was_true == True:
            self.cls_vec.requires_grad_(False)
        novel_clip_num = max(0, num_novel_classes)
        #print("novel_clip_num", novel_clip_num)
        resize_num = novel_clip_num +self.num_classes_base
        old_num = self.num_classes_base + self.num_classes_novel
        #print(self.num_classes_base, self.num_classes_novel, old_num, resize_num)
        if resize_num != old_num:
            num_to_copy = min(old_num, resize_num)
            #print("num_to_copy", num_to_copy)
            cls_vec_new = torch.randn(resize_num, self.dim).to(device) * self.sdev 
            cls_vec_new.requires_grad_(False)
            cls_vec_new[:num_to_copy,:] = self.cls_vec[:num_to_copy,:]
            self.cls_vec = Parameter(cls_vec_new, requires_grad=grad_was_true)
            if grad_was_true == True:
                self.cls_vec.requires_grad_(True)
            self.num_classes_novel = novel_clip_num
        return novel_clip_num
        
        
    def set_include_idxs(self, idxs):
        self.include_idxs = idxs

    def calc_mask(self, to_mask):
        cur_mask = torch.ones_like(to_mask, requires_grad = False)
        if len(self.include_idxs) > 0:
            cur_mask[self.include_idxs] = 1.
        if len(self.exclude_idxs) > 0:
            cur_mask[self.exclude_idxs] = 0.
        return cur_mask

    def apply_mask(self, to_mask, cur_mask):
        ret = torch.mul(to_mask, cur_mask)
        return ret



    def calc_w_att(self, z_arr):
        #z_arr = (k_shot, dim)
        zq = torch.matmul(z_arr, self.phi_q) # (k_shot, dim) x (dim, dim) = (k_shot, dim)
        # "spiked" cos sim, gives sim scores across rows for each k_shot input
        cur_csim = self.gamma * self.cos_sim(zq, self.k_b) # (k_shot, dim) x (dim, nb) = (k_shot, nb)
        # softmax over the base classes (dim = 1)
        cur_smax = self.attn_smax(cur_csim)
        attended = None
        # each row sums over base classes for each k shot input
        if self.train_phase == TrainPhase.base_weightgen:
            # use the hacky pseudo copy of base vectors
            attended = torch.matmul(cur_smax, self.cls_vec_pseudo)
        else:
            # just use the normal base classifier vectors since no training
            attended = torch.matmul(cur_smax, self.cls_vec[:self.num_classes_base])
        kshot_mean = torch.mean(attended, dim=0) # mean over all kshot inputs
        return kshot_mean

    def calc_w_n_plus_1(self, z_arr):
        z_avg = torch.mean(z_arr, dim=0)
        w_att = self.calc_w_att(z_arr)
        cur_wn = torch.mul(self.phi_avg, z_avg) + torch.mul(self.phi_att, w_att)
        return cur_wn
        
    def set_pseudonovel_vec(self, k_novel_idx, k_novel_ft):

        #self.phi_avg = Parameter(torch.randn(dim)*self.sdev) # idea from (3)ish
        #self.phi_att = Parameter(torch.randn(dim)*self.sdev) # idea from (3)ish
        #self.phi_q = Parameter(nn.init.xavier_normal_(torch.zeros(dim,dim))) #idea from (3)ish
        #k_b = torch.randn(num_classes_base, dim) * self.sdev # copying init of cls_vec idea from (3)
        #self.k_b = Parameter(k_b)
        cur_wn = self.calc_w_n_plus_1(k_novel_ft)

        if self.train_phase == TrainPhase.base_weightgen:
            self.cls_vec_copy[k_novel_idx] = cur_wn
        else:
            self.cls_vec[k_novel_idx] = cur_wn

   
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

    def calc_w_n_plus_1_2(self, zarr, zavg, watt):
        # z_arr should be (k, dim)
        w_n_plus_1 = torch.mul(self.phi_avg, zavg) + torch.mul(self.phi_att, watt)
        return w_n_plus_1
   

    def weightgen_train_enable(self, to_enable): 
        self.k_b.requires_grad_(to_enable)
        self.phi_avg.requires_grad_(to_enable)
        self.phi_att.requires_grad_(to_enable)
        self.phi_q.requires_grad_(to_enable)
        try:
            self.cls_vec_copy.requires_grad_(to_enable)
        except:
            pass


    def calc_pseudonovel_vecs(self, zarrs, zavgs, zclasses, watt):
        for i in range(zarrs.shape[0]):
            self.cls_vec_copy[zclasses[i]] = self.calc_w_n_plus_1_2(zarrs[i], zavgs[i], watt[i])

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
            self.cls_vec_copy[cur_idx] = cur_wn
            last_sz = cur_sz

    def forward(self, ipt):
        # input should be (bs, num_channels)
        # return classifier cosine similarity scores
        ret = None
        if self.train_phase == TrainPhase.base_init:
        #idea from (3)
            cur_mask = self.calc_mask(self.cls_vec)
            if self.cls_fn == 'cos_sim':
                ret = self.tau * self.cos_sim(ipt,self.apply_mask(self.cls_vec, cur_mask))
            else:
                ret = self.rev_euc_dist(ipt,self.apply_mask(self.cls_vec, cur_mask))
            #ret = self.cos_sim(ipt,self.apply_mask(self.cls_vec, cur_mask))
        elif self.train_phase == TrainPhase.base_weightgen:
            if self.cls_fn == 'cos_sim':
                ret = self.tau * self.cos_sim(ipt,self.cls_vec_copy)
            else:
                ret = self.rev_euc_dist(ipt,self.cls_vec_copy)
            #ret = self.cos_sim(ipt,self.cls_vec_copy)
        else:
            if self.cls_fn == 'cos_sim':
                ret = self.tau * self.cos_sim(ipt, self.cls_vec)
            else:

                ret = self.rev_euc_dist(ipt,self.cls_vec)
            #ret = self.cos_sim(ipt, self.cls_vec)
        #print(ret)
        return ret
        # should be (bs, num_channels) x (num_channels, num_classes_base) = (bs, num_classes_base)
