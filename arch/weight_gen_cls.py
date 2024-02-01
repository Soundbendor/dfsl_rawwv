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
    def __init__(self, num_classes = 50, dim=512, seed=3, base_idxs = [], novel_idxs = [], exclude_idxs = [], train_phase=TrainPhase.base_init):
        super().__init__()
        torch.manual_seed(seed)

        self.train_phase = train_phase
        self.num_classes = num_classes
        self.dim = dim
        self.base_classes = base_idxs
        self.novel_classes = np.setdiff1d(np.arange(self.num_classes,dtype=int), base_idxs)
        sdev = torch.sqrt(torch.tensor(2.0/dim)) # from (3)
        cls_vec = torch.randn(num_classes,dim) * sdev # from(3)
        #cls_vec = torch.zeros(num_classes,dim)
        #self.cls_vec = Parameter(nn.init.xavier_normal_(cls_vec)) # idea from (3)
        self.cls_vec = Parameter(cls_vec) # idea from (3)
        self.num_base_classes = 0
        self.num_novel_classes = 0
        #from (2)
        tau = torch.tensor(10.)
        gamma = torch.tensor(10.)
        self.tau = Parameter(tau)
        self.gamma = Parameter(gamma)
        self.phi_avg = Parameter(torch.randn(dim)*sdev) # idea from (3)ish
        self.phi_att = Parameter(torch.randn(dim)*sdev) # idea from (3)ish
        self.phi_q = Parameter(nn.init.xavier_normal_(torch.zeros(dim,dim))) #idea from (3)ish
        k_b = torch.randn(num_classes, dim) * sdev # copying init of cls_vec idea from (3)
        self.k_b = Parameter(k_b)
        self.include_idxs = []
        self.exclude_idxs = []
        self.k_b.requires_grad_(False)
        self.phi_avg.requires_grad_(False)
        self.phi_att.requires_grad_(False)
        self.phi_q.requires_grad_(False)
        self.attn_smax = nn.Softmax(dim=0) #used to take attention over softmax for weight gen

    def set_train_phase(self, cur_tph):
        self.train_phase = cur_tph
        if cur_tph != TrainPhase.base_weightgen:
            self.k_b.requires_grad_(False)
            self.phi_avg.requires_grad_(False)
            self.phi_att.requires_grad_(False)
            self.phi_q.requires_grad_(False)
            try:
                del self.cls_vec_copy
                del self.k_b_copy
                del self.cls_vec_attval
                self.cls_vec_copy = None
                self.cls_vec_attval = None
                self.k_b_copy = None
            except:
                pass
        else:
            self.k_b.requires_grad_(True)
            self.phi_avg.requires_grad_(True)
            self.phi_att.requires_grad_(True)
            self.phi_q.requires_grad_(True)


    def clear_exclude_idxs(self):
        self.exclude_idxs.clear()

    def set_base_class_idxs(self, idxs):
        self.base_classes = idxs
        self.num_base_classes = len(idxs)
        self.novel_classes = np.setdiff1d(np.arange(self.num_classes,dtype=int), idxs)
        self.num_novel_classes = self.novel_classes.shape[0]
        self.k_b.requires_grad_(False)
        self.k_b[self.novel_classes] = 0.
        self.k_b.requires_grad_(True)

    def cos_sim(self, ipt1, ipt2):
        ret = torch.matmul(nn.functional.normalize(ipt1,dim=1), nn.functional.normalize(ipt2, dim=1).T)
        return ret
   
    def append_include_idxs(self, idxs):
        self.include_idxs += idxs
        #sorted(self.include_idxs)

    def set_exclude_idxs(self, exclude_idxs):
        self.exclude_idxs = exclude_idxs
        try:
            del self.cls_vec_copy
            del self.cls_vec_attval
            del self.k_b_copy
        except:
            pass
        self.cls_vec_copy = self.cls_vec.clone().detach()
        self.cls_vec_copy.requires_grad_(False)
        self.cls_vec_copy[exclude_idxs] = 0.
        self.cls_vec_attval = self.cls_vec_copy.clone().detach()
        self.cls_vec_attval.requires_grad_(False)

        self.k_b_copy = self.k_b.clone().detach()
        self.k_b_copy[exclude_idxs] = 0.
        self.k_b_copy.requires_grad_(True)
        self.k_b.requires_grad_(False)

    def reset_copies(self):
        try:
            del self.cls_vec_copy
            del self.cls_vec_attval
        except:
            pass
        self.k_b_copy.requires_grad_(False)
        self.k_b_copy[self.exclude_idxs] = self.k_b[self.exclude_idxs]
        self.k_b = nn.Parameter(self.k_b_copy.clone().detach())
        self.k_b.requires_grad_(True)
        try:
            del self.k_b_copy
        except:
            pass



    def set_include_idxs(self, idxs):
        self.include_idxs = idxs

    def calc_mask(self, to_mask):
        cur_mask = torch.zeros_like(to_mask, requires_grad = False)
        if len(self.base_classes) > 0:
            cur_mask[self.base_classes] = 1.
        if len(self.include_idxs) > 0:
            cur_mask[self.include_idxs] = 1.
        if len(self.exclude_idxs) > 0:
            cur_mask[self.exclude_idxs] = 0.
        return cur_mask

    def apply_mask(self, to_mask, cur_mask):
        ret = torch.mul(to_mask, cur_mask)
        return ret

    def calc_w_att(self, z_arr):
        # z_arr should be (k, dim)
        att1 = torch.matmul(self.phi_q, z_arr.T) # should be (dim,dim) x (dim, k) = (dim, k)
        #print(self.k_b.shape, att1.shape, self.phi_q.shape, z_arr.shape)
        #print(att_out.shape)
        multed = None
        if self.train_phase != TrainPhase.base_weightgen:
            cur_mask = self.calc_mask(self.k_b)
            att_out = self.gamma * self.cos_sim(self.apply_mask(self.k_b, cur_mask), att1.T) # (num_classes, dim) x (dim,k) = (num_classes, k)
            multed = torch.matmul(self.attn_smax(att_out).T, self.cls_vec_attval) # (k, num_classes) x (num_classes, dim) = (k, dim)
        else:
            att_out = self.gamma * self.cos_sim(self.k_b_copy, att1.T) # (num_classes, dim) x (dim,k) = (num_classes, k)
            multed = torch.matmul(self.attn_smax(att_out).T, self.cls_vec_attval) # (k, num_classes) x (num_classes, dim) = (k, dim)
        #print(multed.shape)
        #ret = (multed/self.num_base_classes).mean(dim=0) # div by num_base_classes to simulate taking mean, then take actual mean dim = 0 for mean across k
        ret = torch.mean(multed,dim=0) #take actual mean dim = 0 for mean across k
        return ret

    def calc_w_n_plus_1(self, z_arr):
        # z_arr should be (k, dim)
        z_arr.requires_grad_(False)
        z_avg = torch.mean(z_arr,dim=0) # should be (dim)
        z_avg.requires_grad_(False)
        w_att = self.calc_w_att(z_arr) # should be (dim)
        w_n_plus_1 = torch.mul(self.phi_avg, z_avg) + torch.mul(self.phi_att, w_att)
        return w_n_plus_1
    
    def set_pseudonovel_vec(self, k_novel_idx, k_novel_ft):
        # k_novel_ft should be of size (k_novel, dim)
        #k_novel_ft.requires_grad_(False)
        cur_wn = self.calc_w_n_plus_1(k_novel_ft)
        self.cls_vec_copy[k_novel_idx] = cur_wn
    
    def freeze_classifier(self):
        self.cls_vec.requires_grad_(False)
        self.tau.requires_grad_(False)

    def unfreeze_classifier(self):
        self.cls_vec.requires_grad_(True)
        self.tau.requires_grad_(True)
 
    def calc_w_n_plus_1_2(self, zarr, zavg, watt):
        # z_arr should be (k, dim)
        w_n_plus_1 = torch.mul(self.phi_avg, zavg) + torch.mul(self.phi_att, watt)
        return w_n_plus_1
    

    def calc_pseudonovel_vecs(self, zarrs, zavgs, zclasses, watt):
        for i in range(zarrs.shape[0]):
            self.cls_vec_copy[zclasses[i]] = self.calc_w_n_plus_1_2(zarrs[i], zavgs[i], watt[i])

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
        
        if self.train_phase == TrainPhase.base_init:
        #idea from (3)
            cur_mask = self.calc_mask(self.cls_vec)
            ret = self.tau * self.cos_sim(ipt,self.apply_mask(self.cls_vec, cur_mask))
            #ret = self.cos_sim(ipt,self.apply_mask(self.cls_vec, cur_mask))
            return ret
        elif self.train_phase == TrainPhase.base_weightgen:
            ret = self.tau * self.cos_sim(ipt,self.cls_vec_copy)
            #ret = self.cos_sim(ipt,self.cls_vec_copy)
            return ret
        else:
            ret = self.tau * self.cos_sim(ipt, self.cls_vec)
            #ret = self.cos_sim(ipt, self.cls_vec)
            return ret
        # should be (bs, num_channels) x (num_channels, num_classes) = (bs, num_classes)
