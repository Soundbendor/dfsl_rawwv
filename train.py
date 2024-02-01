import numpy as np
import torch
import torchaudio as TA
from torch import nn
from torch.utils.data import DataLoader,Subset
from torch import cuda
from arch.sampcnn_model import SampCNNModel
from ds.esc50 import ESC50 
import os
import argparse
import time
import contextlib
from util.types import BatchType,TrainPhase
import util.results as UR 
import util.metrics as UM
import util.nep as UN
import util.globals as UG
import util.parser as UP
from distutils.util import strtobool
import neptune
from ast import literal_eval as make_tuple

#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479
# (4) Wang, Y., Bryan, N. J., Cartwright, M., Bello, J. P., and Salamon, J. (2021a). Few-Shot Continual Learning for Audio Classification. ICASSP 2021 - 2021 IEEE International Conference on Acoustic, Speech and Signal Processing, 321-325. https://doi.org/10.1109/ICASSP39728.2021.9413584.

# ESC50: 30 (base)-10 novel(val) - 10 novel(test) class split
# 24 (training):8(validation):8(testing) sample split

UG.DEF_DATADIR = os.path.join(os.sep, 'media', 'dxk', 'tosh_ext', 'ds', 'ESC-50-master') 
UG.DEF_SAVEDIR = os.path.join(os.sep, 'media', 'dxk', 'tosh_ext', 'fscil', 'dfsl_rawwv', 'save') 
#UG.DEF_SAVEDIR = os.path.join(os.path.split(__file__)[0], "save")
UG.DEF_GRAPHDIR = os.path.join(os.path.split(__file__)[0], "graph")
UG.DEF_RESDIR = os.path.join(os.path.split(__file__)[0], "res")
UG.DEF_NEP_API = os.path.join(os.path.split(__file__)[0], "nepapi.txt")
UG.DEF_SEED = 3

#torch.autograd.set_detect_anomaly(True)

def make_folder(cur_arg, cur_dir):
    if os.path.exists(cur_dir) != True and cur_arg == True:
        os.makedirs(cur_dir)


def runner(model, expr_idx = 0, train_phase = TrainPhase.base_init, seed=UG.DEF_SEED, sr = 16000, max_samp = 118098, max_rng=10000, lr = 1e-4, bs=4, label_smoothing = 0.0, graph_dir = UG.DEF_GRAPHDIR, save_dir = UG.DEF_SAVEDIR, res_dir = UG.DEF_RESDIR, data_dir = UG.DEF_DATADIR, epochs=1, save_ivl=0, num_classes_total = 50, use_class_weights = False, to_print=True, to_time = True, to_graph=True, to_res = True, device='cpu', nep=None):
    rng = np.random.default_rng(seed=seed)
    cur_seed = rng.integers(0,max_rng,1)[0]
    torch.manual_seed(seed)
    cur_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    cur_optim = torch.optim.Adam(model.parameters(), lr=lr)
    class_order = np.arange(0,num_classes_total) # order of classes
    rng.shuffle(class_order) # shuffle classes
    base_classes = class_order[:30]
    novelval_classes = class_order[30:40]
    noveltest_classes = class_order[40:]
    fold_order = np.arange(1,6) # order of folds
    rng.shuffle(fold_order) # shuffle folds to group folds other than sequentially
    training_folds = fold_order[:3]
    valid_folds = fold_order[3:4]
    test_folds = fold_order[4:]
    base_train_data = ESC50(folds=training_folds, classes=base_classes, k=24, srate=sr, samp_sz=max_samp, basefolder = data_dir, seed = cur_seed)
    base_valid_data = ESC50(folds=valid_folds, classes=base_classes, k=8, srate=sr, samp_sz=max_samp, basefolder = data_dir, seed = cur_seed)
    base_test_data = ESC50(folds=test_folds, classes=base_classes, k=8, srate=sr, samp_sz=max_samp, basefolder = data_dir, seed = cur_seed)
    novelval_train_data = ESC50(folds=training_folds, classes=novelval_classes, k=24, srate=sr, samp_sz=max_samp, basefolder = data_dir, seed = cur_seed)
    novelval_valid_data = ESC50(folds=valid_folds, classes=novelval_classes, k=8, srate=sr, samp_sz=max_samp, basefolder = data_dir, seed = cur_seed)
    novelval_test_data = ESC50(folds=test_folds, classes=novelval_classes, k=8, srate=sr, samp_sz=max_samp, basefolder = data_dir, seed = cur_seed)
    noveltest_train_data = ESC50(folds=training_folds, classes=noveltest_classes, k=24, srate=sr, samp_sz=max_samp, basefolder = data_dir, seed = cur_seed)
    noveltest_valid_data = ESC50(folds=valid_folds, classes=noveltest_classes, k=8, srate=sr, samp_sz=max_samp, basefolder = data_dir, seed = cur_seed)
    noveltest_test_data = ESC50(folds=test_folds, classes=noveltest_classes, k=8, srate=sr, samp_sz=max_samp, basefolder = data_dir, seed = cur_seed)

    model.classifier.set_base_class_idxs(base_train_data.get_class_idxs())
    if use_class_weights == True and train_phase == TrainPhase.base_init:
        cur_loss.weight = torch.tensor(base_train_data.class_prop)
    print("~~~~~")
    print(f"Running Expr {expr_idx} with epochs: {epochs}, bs:{bs}, lr:{lr}\n-----")
    print(f"Training Phase: {train_phase.name}, Printing: {to_print}, Save Model Interval: {save_ivl}, Graphing: {to_graph}, Saving Results: {to_res}")
    print("~~~~~")
    if train_phase == TrainPhase.base_init:
        base_init_trainer(model,cur_loss, cur_optim, base_train_data,base_valid_data, expr_idx= expr_idx, epochs=epochs, lr=lr, bs=bs, save_ivl=save_ivl, num_classes_total = num_classes_total, save_dir=save_dir, res_dir = res_dir, to_print=to_print, to_time=to_time, to_graph=to_graph, to_res=to_res, graph_dir = graph_dir, nep=nep,device=device)
        tester(model,cur_loss,base_test_data, expr_idx= expr_idx, bs=bs, num_classes_total = num_classes_total, res_dir=res_dir, graph_dir = graph_dir, to_print=to_print, to_time=to_time, to_graph=to_graph, to_res=to_res,device=device,pretrain=(train_phase != TrainPhase.base_init),nep=nep)

    if train_phase == TrainPhase.base_weightgen:
        base_weightgen_trainer(model, cur_loss, cur_optim, base_train_data, base_valid_data, lr=lr, bs = bs, epochs = epochs, save_ivl = save_ivl, save_dir = save_dir, res_dir = res_dir, graph_dir = graph_dir, device = device, expr_idx = expr_idx, num_classes_total = num_classes_total, to_print = to_print, to_time = to_time, to_graph = to_graph, to_res = to_res, rng = rng, k_novel = 5, k_samp = 4, base_classes = base_classes, nep=nep)
 

def loss_printer(epoch_idx, batch_idx, cur_loss, loss_type=BatchType.train, to_print = True):
    if to_print == True:
        cur_str = f"{loss_type.name} loss ({epoch_idx},{batch_idx}): {cur_loss}"
        print(cur_str)


def batch_handler(model, dloader, cur_losser, cur_opter=None, batch_type = BatchType.train, train_phase=TrainPhase.base_weightgen, device='cpu', bs=4, epoch_idx=0, num_classes_total = 50, to_print=True, to_time = False):
    #time_batch = []
    loss_batch = []
    curmet = UM.metric_creator(num_classes=num_classes_total)
    #train = not (cur_opter is None)
    train = batch_type.name == 'train'
    time_start = -1
    time_last = -1
    if to_time == True:
        time_start = time.time()
        #time_last = time_start
    if train == False:
        model.eval()
    with (torch.no_grad() if train == False else contextlib.nullcontext()):
        for batch_idx, (ci,cl) in enumerate(dloader):
            pred = model(ci.to(device))
            #print(ci.shape)
            #print(cl.shape)
            cur_loss = cur_losser(pred, cl.to(device))
            loss_item = cur_loss.item()
            if train ==True and train_phase != TrainPhase.base_weightgen:
                cur_loss.backward()
                cur_opter.step()
                cur_opter.zero_grad()
            UM.metric_updater (curmet, pred, cl)
            if to_print == True:
                loss_printer(epoch_idx, batch_idx, loss_item, loss_type=batch_type, to_print = to_print )
            loss_batch.append(loss_item)
            """
            if to_time == True:
                time_finish = time.time()
                batch_times.append(time_finish - time_last)
                time_last = time_start
            """
        if train == True and train_phase == TrainPhase.base_weightgen:
            cur_loss.backward()
            cur_opter.step()
            cur_opter.zero_grad()


    time_avg = -1
    time_batch_overall = -1
    if to_time == True:
        #time_avg = np.mean(time_batch)
        time_batch_overall = time.time() - time_start
        time_avg = time_batch_overall/bs
    loss_avg = np.mean(loss_batch)
    computedmet = UM.metric_compute(curmet)
    if to_print == True:
        UM.metric_printer(computedmet) 
        if to_time == True:
            time_str = f"+ Avg Time: {time_avg}, Overall Time: {time_batch_overall}"
            print(time_str)
    ret = {"epoch_idx": epoch_idx, "batch_type": batch_type.name,  "loss_avg": loss_avg, "time_avg": time_avg}
    ret.update(computedmet)
    """
    ret = {"epoch_idx": epoch_idx, "batch_type": batch_type.name, "epoch_avg_ap": ap_avg,
            "epoch_avg_loss": loss_avg, "epoch_avg_time": time_avg, "epoch_avg_acc1": acc1_avg}
    """

    return ret

def model_saver(cur_model, save_dir=UG.DEF_SAVEDIR, epoch_idx=0, expr_idx = 0, mtype="embedder"):
    save_str=f"{expr_idx}-sampcnn_{mtype}_{epoch_idx}-model.pth"
    outpath = os.path.join(save_dir, save_str)
    cdict = None
    if mtype=="embedder":
        cdict = cur_model.embedder.state_dict()
    else:
        cdict = cur_model.classifier.state_dict()
    torch.save(cdict, outpath)


def tester(model, cur_loss, test_data, bs = 4, res_dir = UG.DEF_RESDIR, graph_dir = UG.DEF_GRAPHDIR, device='cpu', expr_idx = 0, num_classes_total = 50, to_print = True, to_time = True, to_graph = True, to_res = True, pretrain = False,nep=None):
    test_dload = DataLoader(test_data, shuffle=True, batch_size = bs)

    confmat_path = ""
    if to_print == True:
        print(f"\n Testing\n ==========================")
    res_test = batch_handler(model, test_dload, cur_loss, cur_opter=None, batch_type = BatchType.test, device=device, epoch_idx=-1, bs=bs, num_classes_total=num_classes_total, to_print=to_print, to_time = to_time)
    if to_res == True:
        UR.res_csv_appender(res_test, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=-1, batch_type=BatchType.test, expr_name="sampcnn_base", pretrain=pretrain)
    if to_graph == True:
        confmat_path = UR.plot_confmat(res_test['confmat'],dest_dir=graph_dir, t_ph = TrainPhase.base_init, expr_idx=expr_idx, expr_name="sampcnn_base")

    if nep != None:
        UN.nep_batch_parser(nep, res_test,batch_type=BatchType.test, train_phase = TrainPhase.base_init)
        if len(confmat_path) > 0:
            UN.nep_confmat_upload(nep,confmat_path ,batch_type=BatchType.test, train_phase = TrainPhase.base_init)



def base_weightgen_trainer2(model, cur_loss, cur_optim, train_data, valid_data, lr=1e-4, bs = 4, epochs = 1, save_ivl = 0, save_dir = UG.DEF_SAVEDIR, res_dir = UG.DEF_RESDIR, graph_dir = UG.DEF_GRAPHDIR, device = 'cpu', expr_idx = 0, num_classes_total = 50, to_print = True, to_time = True, to_graph = True, to_res = True, rng = None, k_novel = 5, k_samp = 4, base_classes = []):
    model.set_train_phase(TrainPhase.base_weightgen)
    model.zero_grad()
    cur_optim.zero_grad()

    res_train = []
    res_valid = []
    k_dim = 512
    valid_dload = DataLoader(valid_data, shuffle=True, batch_size = bs)
    if rng == None:
        rng = np.random.default_rng(seed=UG.DEF_SEED)
    for epoch_idx in range(epochs):
        cur_novel = rng.choice(base_classes, size=k_novel, replace=False)
        cur_base = np.setdiff1d(base_classes,cur_novel)
        # first sample from each class the examples to use to generate the classification vectors
        # and then some how sample other examples for t_novel and t_test (see gidaris supp)
        # and then do backprop to learn phi matrices and vectors
        # one idea is to sample t_novel from the remaining examples from the training set to there's only 8 vectors per class (wang doesn't specify where these come from)
        # another idea is to sample from validation set but then that might be information leakage
        # in any case, T_novel should be the same as t_base per class (ideally)
        # and we are only SIMULATING few-shot so it doesn't actually have to be fewshot
        # for learning novel classes, phi matrices should be fixed
       
        model.set_exclude_idxs(cur_novel)
        test_k = np.array([],dtype=int)
        test_b = np.array([], dtype=int)
        test_num = np.inf 
        model.init_zarr(k_novel, k_samp, k_dim,device=device)

        for novel_cls in cur_novel: # sampling for classification vectors
            cur_k_idxs = train_data.get_class_ex_idxs(novel_cls)
            wg_k = rng.choice(cur_k_idxs, size=k_samp, replace=False)
            unsampled = np.setdiff1d(cur_k_idxs, wg_k)
            cur_subset = Subset(train_data, wg_k)
            subset_dl = DataLoader(cur_subset, batch_size=k_samp, shuffle=False)
            #print(wg_k, unsampled)
            test_k = np.append(test_k,unsampled)
            test_num = min(test_num, unsampled.shape[0])
            for ci,cl in subset_dl:
                model.set_zarr(ci.to(device), novel_cls)
        model.calc_pseudonovel_vecs()
        for base_cls in cur_base:
            cur_b_idxs = train_data.get_class_ex_idxs(base_cls)
            base_test = rng.choice(cur_b_idxs, size=test_num, replace=False)
            test_b = np.append(test_b, base_test)
        test_all = np.append(test_k,test_b)
        test_set = Subset(train_data, test_all)
        test_dl = DataLoader(test_set, batch_size=bs, shuffle=True)
         
        if to_print == True:
            print(f"\nEpoch {epoch_idx}\n ==========================")
        res_wgen = batch_handler(model, test_dl, cur_loss, cur_opter=cur_optim, batch_type = BatchType.train, device=device, epoch_idx=epoch_idx, train_phase=TrainPhase.base_weightgen, bs=bs, num_classes_total = num_classes_total, to_print=to_print, to_time = to_time)
        model.reset_copies()
        if to_res == True:
            UR.res_csv_appender(res_wgen, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=epoch_idx, batch_type=BatchType.train, expr_name="sampcnn_wgen")
        if save_ivl > 0:
            if ((epoch_idx +1) % save_ivl == 0 and epoch_idx != 0) or epoch_idx == (epochs-1):
                #model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_idx=expr_idx, mtype="embedder")
                model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_idx=expr_idx, mtype="classifier")

        if nep != None:
            UN.nep_batch_parser(nep, res_train,batch_type=BatchType.train, train_phase = TrainPhase.base_weightgen)
        res_valid = batch_handler(model, valid_dload, cur_loss, cur_opter=None, batch_type = BatchType.valid, device=device, epoch_idx=epoch_idx, bs=bs, to_print=to_print, to_time = to_time)
        if to_res == True:
            UR.res_csv_appender(res_valid, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=epoch_idx, batch_type=BatchType.valid, expr_name="sampcnn_wgen", pretrain = False)
        res_train_batches.append(res_train)
        res_valid_batches.append(res_valid)

        if nep != None:
            UN.nep_batch_parser(nep, res_valid,batch_type=BatchType.valid, train_phase = TrainPhase.base_weightgen)
    if to_graph == True:
        UR.train_valid_grapher(res_train_batches, res_valid_batches, dest_dir="graph", graph_key="loss_avg", expr_idx=expr_idx, expr_name="sampcnn_wgen")
        UR.train_valid_grapher(res_train_batches, res_valid_batches, dest_dir="graph", graph_key="time_avg", expr_idx=expr_idx, expr_name="sampcnn_wgen")


       


def base_weightgen_trainer(model, cur_loss, cur_optim, train_data, valid_data, lr=1e-4, bs = 4, epochs = 1, save_ivl = 0, save_dir = UG.DEF_SAVEDIR, res_dir = UG.DEF_RESDIR, graph_dir = UG.DEF_GRAPHDIR, device = 'cpu', expr_idx = 0, num_classes_total = 50, to_print = True, to_time = True, to_graph = True, to_res = True, rng = None, k_novel = 5, k_samp = 5, max_samp = 24, total_novel_samp = 100, total_base_samp = 100, base_classes = [],nep = None):
    model.set_train_phase(TrainPhase.base_weightgen)
    model.zero_grad()
    cur_optim.zero_grad()
    num_base_classes = len(base_classes)
    num_rest = num_base_classes - k_novel
    # number of unsampled from novel to take for training against
    num_novel_unsampled = int((total_novel_samp - (k_novel * k_samp))/k_novel)
    # number of unsampled (all of them are) to take for training against (from rest of base classes)
    num_base_unsampled = int(total_base_samp/num_rest)
    res_wgen_batches = []
    res_valid_batches = []
    valid_dload = DataLoader(valid_data, shuffle=True, batch_size = bs)
    if rng == None:
        rng = np.random.default_rng(seed=UG.DEF_SEED)
    for epoch_idx in range(epochs):
        cur_novel = rng.choice(base_classes, size=k_novel, replace=False)
        cur_base = np.setdiff1d(base_classes,cur_novel)
        # first sample from each class the examples to use to generate the classification vectors
        # and then some how sample other examples for t_novel and t_test (see gidaris supp)
        # and then do backprop to learn phi matrices and vectors
        # one idea is to sample t_novel from the remaining examples from the training set to there's only 8 vectors per class (wang doesn't specify where these come from)
        # another idea is to sample from validation set but then that might be information leakage
        # in any case, T_novel should be the same as t_base per class (ideally)
        # and we are only SIMULATING few-shot so it doesn't actually have to be fewshot
        # for learning novel classes, phi matrices should be fixed
       
        model.set_exclude_idxs(cur_novel)
        test_k = np.array([],dtype=int)
        test_b = np.array([], dtype=int)
        for novel_cls in cur_novel: # sampling for classification vectors
            cur_k_idxs = train_data.get_class_ex_idxs(novel_cls)
            wg_k = rng.choice(cur_k_idxs, size=k_samp, replace=False)
            unsampled = np.setdiff1d(cur_k_idxs, wg_k)
            cur_subset = Subset(train_data, wg_k)
            subset_dl = DataLoader(cur_subset, batch_size=k_samp, shuffle=False)
            #print(wg_k, unsampled)
            unsampled_samp = rng.choice(unsampled, size=num_novel_unsampled, replace=False)
            test_k = np.append(test_k,unsampled_samp)
            for ci,cl in subset_dl:
                model.set_pseudonovel_vec(novel_cls, ci.to(device))
        for base_cls in cur_base:
            cur_b_idxs = train_data.get_class_ex_idxs(base_cls)
            base_test = rng.choice(cur_b_idxs, size=num_base_unsampled, replace=False)
            test_b = np.append(test_b, base_test)
        test_all = np.append(test_k,test_b)
        test_set = Subset(train_data, test_all)
        test_dl = DataLoader(test_set, batch_size=bs, shuffle=True)
         
        if to_print == True:
            print(f"\nEpoch {epoch_idx}\n ==========================")
        res_wgen = batch_handler(model, test_dl, cur_loss, cur_opter=cur_optim, batch_type = BatchType.train, device=device, epoch_idx=epoch_idx, train_phase = TrainPhase.base_weightgen, bs=bs, num_classes_total = num_classes_total, to_print=to_print, to_time = to_time)
        print("got to here")
        if to_res == True:
            UR.res_csv_appender(res_wgen, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=epoch_idx, batch_type=BatchType.train, expr_name="sampcnn_wgen")
        if save_ivl > 0:
            if ((epoch_idx +1) % save_ivl == 0 and epoch_idx != 0) or epoch_idx == (epochs-1):
                #model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_idx=expr_idx, mtype="embedder")
                model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_idx=expr_idx, mtype="classifier")

        if nep != None:
            UN.nep_batch_parser(nep, res_wgen,batch_type=BatchType.train, train_phase = TrainPhase.base_weightgen)
        res_valid = batch_handler(model, valid_dload, cur_loss, cur_opter=None, batch_type = BatchType.valid, device=device, epoch_idx=epoch_idx, bs=bs, to_print=to_print, to_time = to_time)

        model.reset_copies()
        if to_res == True:
            UR.res_csv_appender(res_valid, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=epoch_idx, batch_type=BatchType.valid, expr_name="sampcnn_wgen", pretrain = False)
        res_wgen_batches.append(res_wgen)
        res_valid_batches.append(res_valid)

        if nep != None:
            UN.nep_batch_parser(nep, res_valid,batch_type=BatchType.valid, train_phase = TrainPhase.base_weightgen)
    if to_graph == True:
        UR.train_valid_grapher(res_wgen_batches, res_valid_batches, dest_dir="graph", graph_key="loss_avg", expr_idx=expr_idx, expr_name="sampcnn_wgen")
        UR.train_valid_grapher(res_valid_batches, res_valid_batches, dest_dir="graph", graph_key="time_avg", expr_idx=expr_idx, expr_name="sampcnn_wgen")


       







            


 


def base_init_trainer(model, cur_loss, cur_optim, train_data, valid_data, lr=1e-4, bs = 4, epochs = 1, save_ivl=0, save_dir=UG.DEF_SAVEDIR, res_dir = UG.DEF_RESDIR, graph_dir = UG.DEF_GRAPHDIR, device='cpu', expr_idx = 0, num_classes_total = 50, to_print = True, to_time = True, to_graph = True, to_res = True, nep=None):
    train_dload = DataLoader(train_data, shuffle=True, batch_size = bs)
    valid_dload = DataLoader(valid_data, shuffle=True, batch_size = bs)
    #model.classifier.set_base_class_idxs(train_data.get_class_idxs())
    res_train_batches = []
    res_valid_batches = []
    for epoch_idx in range(epochs):
        if to_print == True:
            print(f"\nEpoch {epoch_idx}\n ==========================")
        res_train = batch_handler(model, train_dload, cur_loss, cur_opter=cur_optim, batch_type = BatchType.train, device=device, train_phase=TrainPhase.base_init, epoch_idx=epoch_idx, bs=bs, num_classes_total = num_classes_total, to_print=to_print, to_time = to_time)
        if to_res == True:
            UR.res_csv_appender(res_train, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=epoch_idx, batch_type=BatchType.train, expr_name="sampcnn_base")
        if save_ivl > 0:
            if ((epoch_idx +1) % save_ivl == 0 and epoch_idx != 0) or epoch_idx == (epochs-1):
                model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_idx=expr_idx, mtype="embedder")
                model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_idx=expr_idx, mtype="classifier")

        if nep != None:
            UN.nep_batch_parser(nep, res_train,batch_type=BatchType.train, train_phase = TrainPhase.base_init)
        res_valid = batch_handler(model, valid_dload, cur_loss, cur_opter=None, batch_type = BatchType.valid, device=device, train_phase=TrainPhase.base_init, epoch_idx=epoch_idx, bs=bs, to_print=to_print, to_time = to_time)
        if to_res == True:
            UR.res_csv_appender(res_valid, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=epoch_idx, batch_type=BatchType.valid, expr_name="sampcnn_base", pretrain = False)
        res_train_batches.append(res_train)
        res_valid_batches.append(res_valid)

        if nep != None:
            UN.nep_batch_parser(nep, res_valid,batch_type=BatchType.valid, train_phase = TrainPhase.base_init)
    if to_graph == True:
        UR.train_valid_grapher(res_train_batches, res_valid_batches, dest_dir="graph", graph_key="loss_avg", expr_idx=expr_idx, expr_name="sampcnn_base")
        UR.train_valid_grapher(res_train_batches, res_valid_batches, dest_dir="graph", graph_key="time_avg", expr_idx=expr_idx, expr_name="sampcnn_base")

def is_valid_tup(cur_tup):
    return "(" in cur_tup and ")" in cur_tup and "," in cur_tup

if __name__ == "__main__":
    expr_idx = int(time.time() * 1000)
    args = UP.parse_args()
    #print(args)
    make_folder(args.save_ivl > 0, args.save_dir) 
    make_folder(args.to_graph, args.graph_dir) 
    make_folder(args.to_res, args.res_dir) 
    #(num,ksize,stride) = (10,3,3),(2,2,2) gives 236196 which is 15696 extra samples 
    # (includes starting strided conv and following regular conv with strided maxpool)
    # but doesn't include 1 channel conv
    
    #(1) multiplies channels by 2 if 3rd block after strided or if last "config block"
    # also omits stride 1 conv as found in (3)

    # (num, ksize, out_ch, stride)
    strided_list = [(1,3,128,3)]
    #res1_list = [(2,3,128,3), (7,3,256,3),(1,2,256,2), (1,2,512,2)]
    res1_list = []
    #res2_list = [(2,3,128,3), (7,3,256,3),(1,2,512,2)]
    #rese2_list = []
    #rese2_list = [(2,3,128,3), (7,3,256,3),(1,2,512,2)]
    rese2_list = [(2,3,128,3), (7,3,256,3),(1,3,512,3)]
    #se_list = [(2,3,128,3), (7,3,256,3),(1,2,512,2)]
    se_list = []
    res2_list = []
    simple_list = []
    # middle dim according to (1) is same as num channels
    num_classes_total = 50
    device = 'cpu'
    t_ph = TrainPhase.base_init
    #max_samp = 118098
    max_samp = 177147

    if args.train_phase == "base_weightgen":
        t_ph = TrainPhase.base_weightgen
    if torch.cuda.is_available() == True:
        device = 'cuda'

    model = SampCNNModel(in_ch=1, strided_list=strided_list, basic_list=[], res1_list=res1_list, res2_list=res2_list, se_list=se_list, rese1_list=[], rese2_list=rese2_list, simple_list=simple_list, se_dropout=args.se_dropout, res1_dropout=args.res1_dropout, res2_dropout=args.res2_dropout, rese1_dropout=args.rese1_dropout, rese2_dropout=args.rese2_dropout,simple_dropout=args.simple_dropout, se_fc_alpha=args.se_fc_alpha, rese1_fc_alpha=args.rese1_fc_alpha, rese2_fc_alpha=args.rese2_fc_alpha, num_classes=num_classes_total, sr=args.sample_rate, omit_last_relu = args.omit_last_relu, use_prelu = args.use_prelu, se_prelu=args.se_prelu).to(device)

    load_emb = args.load_emb
    load_cls = args.load_cls
    if is_valid_tup(args.load_model_by_tup) == True:
        (expr_idx,num) = make_tuple(args.load_model_by_tup)
        load_cls_fname = f"{expr_idx}-sampcnn_classifier_{num}-model.pth"
        load_emb_fname = f"{expr_idx}-sampcnn_embedder_{num}-model.pth"
        load_emb = os.path.join(args.model_dir, load_emb_fname)
        load_cls = os.path.join(args.model_dir, load_cls_fname)
    if ".pth" in load_emb:
        load_file = load_emb
        if is_valid_tup(args.load_model_by_tup) == False:
            expr_idx = int(load_file.split(os.sep)[-1].split("-")[0])
        t_ph = TrainPhase.base_weightgen
        model.embedder.load_state_dict(torch.load(load_emb))

    if ".pth" in load_cls:
        model.classifier.load_state_dict(torch.load(load_cls))

    settings_dict = {"expr_idx": expr_idx, "lr": args.learning_rate, "bs": args.batch_size,
        "epochs": args.epochs, "sr": args.sample_rate,
        "se_dropout": args.se_dropout,
        "res1_dropout": args.res1_dropout, "res2_dropout": args.res2_dropout,
        "rese1_dropout": args.rese1_dropout, "rese2_dropout": args.res2_dropout,
        "simple_dropout": args.simple_dropout, "use_class_weights": args.use_class_weights,
        "se_fc_alpha": args.se_fc_alpha, "rese1_fc_alpha": args.rese1_fc_alpha, "rese2_fc_alpha": args.rese2_fc_alpha, 
        "label_smoothing": args.label_smoothing, "omit_last_relu": args.omit_last_relu, "use_prelu": args.use_prelu, "se_prelu": args.se_prelu
        }

    if args.to_res == True:
        UR.settings_csv_writer(settings_dict, dest_dir = args.res_dir, expr_idx = expr_idx, expr_name="sampcnn_base")

    nrun = None 
    # NEPTUNE STUFF
    if args.to_nep == True:
        nep_api = ""
        print(f"running neptune: {args.to_nep}")
        with open(UG.DEF_NEP_API, "r") as f:
            nep_api = f.read().strip()


        nrun = neptune.init_run(
            project="Soundbendor/dfsl-rawwv",
            api_token= nep_api,
            capture_hardware_metrics=False,
            )

        nrun["model/params"] = settings_dict

    #print(model.embedder.state_dict())
    runner(model, train_phase = t_ph,expr_idx = expr_idx, lr=args.learning_rate, bs=args.batch_size, epochs=args.epochs, save_ivl = args.save_ivl, sr = args.sample_rate, max_samp = max_samp, use_class_weights = args.use_class_weights, num_classes_total = num_classes_total, label_smoothing = args.label_smoothing,
            res_dir=args.res_dir, save_dir=args.save_dir, to_print=args.to_print, to_time=args.to_time, graph_dir = args.graph_dir, data_dir=args.data_dir, to_graph=args.to_graph, to_res=args.to_res, device=device, nep = nrun)
    if args.to_nep == True:
        nrun.stop()
