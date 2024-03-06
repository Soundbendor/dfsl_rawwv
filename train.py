import numpy as np
import torch
import torchaudio as TA
from torch import nn
from torch.utils.data import DataLoader,Subset
from torch import cuda
from arch.sampcnn_model import SampCNNModel
from arch.cnn14_model import CNN14Model
from ds.esc50 import ESC50, make_esc50_fewshot_tasks 
from ds.tinysol import TinySOL, make_tinysol_fewshot_tasks
import os
import argparse
import time
import contextlib
from util.types import BatchType,TrainPhase,DatasetType,ModelName,DatasetName
import util.results as UR 
import util.metrics as UM
import util.nep as UN
import util.globals as UG
import util.parser as UP
import tomllib
from distutils.util import strtobool
import neptune
from ast import literal_eval as make_tuple
import pandas as pd

#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479
# (4) Wang, Y., Bryan, N. J., Cartwright, M., Bello, J. P., and Salamon, J. (2021a). Few-Shot Continual Learning for Audio Classification. ICASSP 2021 - 2021 IEEE International Conference on Acoustic, Speech and Signal Processing, 321-325. https://doi.org/10.1109/ICASSP39728.2021.9413584.

# ESC50: 30 (base)-10 novel(val) - 10 novel(test) class split
# 24 (training):8(validation):8(testing) sample split

#torch.autograd.set_detect_anomaly(True)

def make_folder(cur_arg, cur_dir):
    if os.path.exists(cur_dir) != True and cur_arg == True:
        os.makedirs(cur_dir)


def runner(model, expr_num = 0, train_phase = TrainPhase.base_init, seed=UG.DEF_SEED, sr = 16000, max_samp = 118098, max_rng=10000, lr = 1e-4, bs=4, label_smoothing = 0.0, graph_dir = UG.DEF_GRAPHDIR, save_dir = UG.DEF_SAVEDIR, res_dir = UG.DEF_RESDIR, data_dir = UG.DEF_ESC50DIR, base_epochs=1, weightgen_epochs = 10, novel_epochs = 10, save_ivl=0, n_way = 5, k_shot = 4, use_class_weights = False, to_print=True, to_time = True, to_graph=True, to_res = True, modelname = ModelName.samplecnn, baseset = DatasetName.esc50, novelset = DatasetName.esc50, device='cpu', multilabel=True, nep=None):
    rng = np.random.default_rng(seed=seed)
    cur_seed = rng.integers(0,max_rng,1)[0]
    torch.manual_seed(seed)
    cur_loss = None
    use_one_hot = multilabel == True
    if multilabel == True:
        cur_loss = nn.BCEWithLogitsLoss()
    else:
        cur_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    cur_optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    esc50path = os.path.join(data_dir, "ESC-50-master")
    esc50_df = pd.read_csv(os.path.join(esc50path, "meta", "esc50.csv"))
    tinysolpath = os.path.join(data_dir, "TinySOL")
    tinysol_df = pd.read_csv(os.path.join(tinysolpath, "TinySOL_metadata.csv"))

   
    esc50_num_classes_base = 30
    esc50_num_classes_valid = 10
    esc50_num_classes_test = 10
    esc50_num_classes_total = esc50_num_classes_base + esc50_num_classes_valid + esc50_num_classes_test
    esc50_class_order = np.arange(0,esc50_num_classes_total) # order of classes
    rng.shuffle(esc50_class_order) # shuffle classes
    esc50_base_classes = esc50_class_order[:esc50_num_classes_base]
    esc50_novelval_classes = esc50_class_order[esc50_num_classes_base: esc50_num_classes_base + esc50_num_classes_valid]
    esc50_noveltest_classes = esc50_class_order[esc50_num_classes_base + esc50_num_classes_valid: esc50_num_classes_total]
    esc50_fold_order = np.arange(1,6) # order of folds
    rng.shuffle(esc50_fold_order) # shuffle folds to group folds other than sequentially
    esc50_training_folds = esc50_fold_order[:3]
    esc50_valid_folds = esc50_fold_order[3:4]
    esc50_test_folds = esc50_fold_order[4:]
    #print("base_train")

    # ~~~~~ DATALOADING ~~~~~
    base_train_data = None
    base_valid_data = None
    base_test_data = None
    novel_val_datas = None
    novel_test_datas = None
    num_classes_base = None
    num_classes_valid = None
    num_classes_test = None
    if baseset == DatasetName.esc50:
        num_classes_base = 30
        base_train_data = ESC50(esc50_df, folds=esc50_training_folds, classes=esc50_base_classes, k_shot=24, srate=sr, samp_sz=max_samp, basefolder = esc50path, seed = cur_seed, label_offset = 0, one_hot = use_one_hot, to_label_tx = True)
        #print("base_valid")
        base_valid_data = ESC50(esc50_df, folds=esc50_valid_folds, classes=esc50_base_classes, k_shot=8, srate=sr, samp_sz=max_samp, basefolder = esc50path, seed = cur_seed, label_offset = 0, one_hot = use_one_hot, to_label_tx = True)
        #print("base_test")
        base_test_data = ESC50(esc50_df, folds=esc50_test_folds, classes=esc50_base_classes, k_shot=8, srate=sr, samp_sz=max_samp, basefolder = esc50path, seed = cur_seed, label_offset = 0, one_hot = use_one_hot, to_label_tx = True)



    # essentially can take from all folds
    # set k to np inf to just take all the possible examples
    if novelset == DatasetName.esc50:
        num_classes_valid = 10
        num_classes_test = 10
        novel_val_datas = make_esc50_fewshot_tasks(esc50_df, folds=esc50_fold_order, classes=esc50_novelval_classes, n_way = n_way, k_shot = np.inf, srate=sr, samp_sz=max_samp, basefolder = esc50path, seed= cur_seed, initial_label_offset = esc50_num_classes_base, one_hot = use_one_hot, to_label_tx = True)
        novel_test_datas = make_esc50_fewshot_tasks(esc50_df, folds=esc50_fold_order, classes=esc50_noveltest_classes, n_way = n_way, k_shot = np.inf, srate=sr, samp_sz=max_samp, basefolder = esc50path, seed= cur_seed, initial_label_offset = esc50_num_classes_base, one_hot = use_one_hot, to_label_tx = True)
    elif novelset == DatasetName.tinysol:
        num_classes_valid = 14
        num_classes_test = 14
        cur_classes = list(range(14))
        cur_n_way = 5
        cur_valid_folds = [0,1]
        cur_test_folds = [2,3]
        cur_valid_k_shot = 19 * len(cur_valid_folds) # 19 is the minimal number of samples for an instrument
        cur_test_k_shot = 19 * len(cur_test_folds) # 19 is the minimal number of samples for an instrument
        # let's just stick with 5 seconds as a nice middle ground
        novel_val_datas = make_tinysol_fewshot_tasks(tinysol_df, folds=cur_valid_folds,classes=cur_classes, n_way = cur_n_way, k_shot=cur_valid_k_shot, srate = sr, samp_sz = max_samp,  basefolder = tinysolpath, seed = cur_seed, initial_label_offset = num_classes_base, one_hot = use_one_hot, to_label_tx = True)
        novel_test_datas = make_tinysol_fewshot_tasks(tinysol_df, folds=cur_test_folds,classes=cur_classes, n_way = cur_n_way, k_shot=cur_test_k_shot, srate = sr, samp_sz = max_samp,  basefolder = tinysolpath, seed = cur_seed, initial_label_offset = num_classes_base, one_hot = use_one_hot, to_label_tx = True)

    if use_class_weights == True and train_phase == TrainPhase.base_init:
        cur_loss.weight = torch.tensor(base_train_data.class_prop)
    print("~~~~~")
    print(f"Running Expr {expr_num} with epochs: ({base_epochs}/{weightgen_epochs}/{novel_epochs}), bs:{bs}, lr:{lr}\n-----")
    print(f"Training Phase: {train_phase.name}, Printing: {to_print}, Save Model Interval: {save_ivl}, Graphing: {to_graph}, Saving Results: {to_res}")
    print("~~~~~")
    if train_phase in [TrainPhase.base_init, TrainPhase.base_trainall, TrainPhase.run_all]:
        base_init_trainer(model,cur_loss, cur_optim, base_train_data,base_valid_data, expr_num= expr_num, epochs=base_epochs, lr=lr, bs=bs, save_ivl=save_ivl, num_classes = num_classes_base, save_dir=save_dir, res_dir = res_dir, to_print=to_print, to_time=to_time, to_graph=to_graph, to_res=to_res, graph_dir = graph_dir, multilabel=multilabel, modelname=modelname, baseset = baseset, novelset = novelset, nep=nep,device=device)
        base_tester(model,cur_loss,base_test_data, expr_num= expr_num, bs=bs, num_classes = num_classes_base, res_dir=res_dir, graph_dir = graph_dir, to_print=to_print, to_time=to_time, to_graph=to_graph, to_res=to_res,device=device,pretrain=(train_phase != TrainPhase.base_init), modelname = modelname, baseset = baseset, novelset = novelset, multilabel=multilabel, nep=nep)

    if train_phase in [TrainPhase.base_weightgen, TrainPhase.base_trainall, TrainPhase.run_all]:
        base_weightgen_trainer(model, cur_loss, cur_optim, base_train_data, base_valid_data, lr=lr, bs = bs, epochs = weightgen_epochs, save_ivl = save_ivl, save_dir = save_dir, res_dir = res_dir, graph_dir = graph_dir, device = device, expr_num = expr_num, multilabel=multilabel,num_classes = num_classes_base, to_print = to_print, to_time = to_time, to_graph = to_graph, to_res = to_res, rng = rng, n_way = n_way, k_shot = k_shot, modelname = modelname, baseset=baseset, novelset=novelset, nep=nep)
 
    if train_phase in [TrainPhase.novel_valid, TrainPhase.run_all]:
        novel_tester(model, cur_loss, base_test_data, novel_val_datas, bs = bs, epochs = novel_epochs, res_dir = res_dir, graph_dir = graph_dir, device = device, expr_num = expr_num, to_print = to_print, to_time = to_time, to_graph = to_graph, to_res = to_res, rng = rng, n_way = n_way, k_shot = k_shot, max_num_test=8,  modelname = modelname, baseset = baseset, novelset = novelset, train_phase=TrainPhase.novel_valid, batch_type=BatchType.test, multilabel = multilabel, nep = nep)
def loss_printer(epoch_idx, batch_idx, cur_loss, loss_type=BatchType.train, to_print = True):
    if to_print == True:
        cur_str = f"{loss_type.name} loss ({epoch_idx},{batch_idx}): {cur_loss}"
        print(cur_str)


def batch_handler(model, dloader_arr, cur_losser, cur_opter=None, batch_type = BatchType.train, train_phase=TrainPhase.base_weightgen, device='cpu', bs=4, epoch_idx=0, num_classes = 50, to_print=True, to_time = False, modelname = ModelName.samplecnn, dsname = DatasetName.esc50, ds_idx = 0, multilabel=True):
    #time_batch = []
    loss_batch = []
    curmet = UM.metric_creator(num_classes=num_classes, multilabel=multilabel)
    #train = not (cur_opter is None)
    train = batch_type.name == 'train'
    time_start = -1
    time_last = -1
    if to_time == True:
        time_start = time.time()
        #time_last = time_start
    if train == False:
        model.eval()
    else:
        model.train()
    #print(f"Before no_grad {model.classifier.cls_vec.requires_grad}")
    with (torch.no_grad() if train == False else contextlib.nullcontext()):
        for dloader in dloader_arr:
            for batch_idx, (ci,cl) in enumerate(dloader):
                #print(ci.shape, cl.shape)
                pred = model(ci.to(device))
                #print("--- predicted ---")
                #print(pred.argmax(dim=1))
                #print("--- true ---")
                #print(cl)
                #print(bs, pred.shape)
                #print(ci.shape)
                #print(cl.shape)
                #print(ci,cl)
                #cur_loss = None
                #cur_loss = cur_losser(pred, cl.to(torch.float).to(device))
                cur_loss = cur_losser(pred, cl.to(device))
                #print(pred.shape)
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
            #del cur_loss
    #print(f"After no_grad {model.classifier.cls_vec.requires_grad}")
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
    ret = {"epoch_idx": epoch_idx, "ds_idx": ds_idx, "dataset": dsname, "batch_type": batch_type.name,  "loss_avg": loss_avg, "time_avg": time_avg}
    ret.update(computedmet)
    if multilabel == True:
        ret["confmat"] = curmet["confmat"]
    """
    ret = {"epoch_idx": epoch_idx, "batch_type": batch_type.name, "epoch_avg_ap": ap_avg,
            "epoch_avg_loss": loss_avg, "epoch_avg_time": time_avg, "epoch_avg_acc1": acc1_avg}
    """

    return ret

def model_saver(cur_model, save_dir=UG.DEF_SAVEDIR, epoch_idx=0, expr_num = 0, modelname = ModelName.samplecnn, baseset=DatasetName.esc50, novelset = DatasetName.esc50, model_idx = 0, mtype="embedder"):
    save_str=f"{expr_num}_{model_idx}-{modelname.name}-{baseset.name}_{mtype}_{epoch_idx}-model.pth"
    outpath = os.path.join(save_dir, save_str)
    cdict = None
    if mtype=="embedder":
        cdict = cur_model.embedder.state_dict()
    else:
        cdict = cur_model.classifier.state_dict()
    torch.save(cdict, outpath)


def base_tester(model, cur_loss, test_data, bs = 4, res_dir = UG.DEF_RESDIR, graph_dir = UG.DEF_GRAPHDIR, device='cpu', expr_num = 0, num_classes = 30, to_print = True, to_time = True, to_graph = True, to_res = True, pretrain = False, modelname = ModelName.samplecnn, baseset = DatasetName.esc50, novelset = DatasetName.esc50,nep=None, multilabel=False):
    model.set_train_phase(TrainPhase.base_test)
    test_dload = DataLoader(test_data, shuffle=True, batch_size = bs)

    confmat_path = ""
    if to_print == True:
        print(f"\n Testing\n ==========================")
    res_test = batch_handler(model, [test_dload], cur_loss, cur_opter=None, batch_type = BatchType.test, device=device, epoch_idx=-1, bs=bs, num_classes=num_classes, to_print=to_print, to_time = to_time, modelname = modelname, dsname=baseset, ds_idx=0, multilabel=multilabel)
    if to_res == True:
        UR.res_csv_appender(res_test, dest_dir=res_dir, expr_num = expr_num, epoch_idx=-1, batch_type=BatchType.test, modelname = modelname, baseset = baseset, novelset = novelset, train_phase = TrainPhase.base_init, pretrain=pretrain)
    if to_graph == True:
        confmat_path = UR.plot_confmat(res_test['confmat'],multilabel=res_test['multilabel'],dest_dir=graph_dir, train_phase = TrainPhase.base_init, expr_num=expr_num, modelname = modelname, baseset=baseset, novelset=novelset)

    if nep != None:
        UN.nep_batch_parser(nep, res_test,batch_type=BatchType.test, train_phase = TrainPhase.base_init, modelname = modelname, ds_type = DatasetType.base, dsname = baseset)
        if len(confmat_path) > 0:
            UN.nep_confmat_upload(nep,confmat_path ,batch_type=BatchType.test, train_phase = TrainPhase.base_init, modelname = modelname, ds_type = DatasetType.base, dsname = baseset)

def novel_tester(model, cur_loss, base_test_data, novel_test_datas, bs = 4, epochs = 1, res_dir = UG.DEF_RESDIR, graph_dir = UG.DEF_GRAPHDIR, device = 'cpu', expr_num = 0, to_print = True, to_time = True, to_graph = True, to_res = True, rng = None, n_way = 5, k_shot = 5, max_num_test=8,  modelname = ModelName.samplecnn, baseset = DatasetName.esc50, novelset = DatasetName.esc50, train_phase=TrainPhase.novel_test, batch_type=BatchType.test, multilabel = True, nep = None):
    # for each class in batch, sample k shot (usually 5) and feed into weight generator
    # then test on novel and base classes
    
    model.eval()
    model.zero_grad()
    model.set_train_phase(train_phase)

    model.freeze_classifier(True)
    base_class_idxs = base_test_data.get_class_idxs()
    num_base = len(base_class_idxs)

    # base_class_results
    res_base_arr = []
    # novel class results, entries will be of increasing length (adding 1 incremental ds each time)
    res_novel_arr = []
    #print("base_classes", base_class_idxs)
    for epoch_idx in range(epochs):
        if to_print == True:
            print(f"\nEpoch {epoch_idx}\n ==========================")
        # do the incremental learning with successive n_way, k_shot novel datasets 
        model.renum_novel_classes(0, device=device)
        num_novel = 0
        #print(f"After renum {model.classifier.cls_vec.requires_grad}")
        novel_test_unsampled = []
        for i,novel_tup in enumerate(novel_test_datas):
            if to_print == True:
                print(f"~~~~~~Novel Task {i}~~~~~~")
            # collect unsampled indices per class for current dataset
            cur_unsamp = []
            (num_novel_to_add, novel_ds) = novel_tup
            num_novel += num_novel_to_add
            model.renum_novel_classes(num_novel,device=device)
            #print(model.classifier.cls_vec.shape)
            #print(f"After Renum {model.classifier.cls_vec.requires_grad}")
            test_b = np.array([], dtype=int)
            test_k = np.array([],dtype=int)

            #unmapped class indices
            novel_class_idxs = novel_ds.get_class_idxs()
            #print("novel_classes_wg", i, novel_class_idxs)
            for novel_class_idx in novel_class_idxs:
                cur_k_idxs = novel_ds.get_class_ex_idxs(novel_class_idx)
                #print("cur_k_idxs", cur_k_idxs)
                # examples for weight generator
                wg_ex = rng.choice(cur_k_idxs, size=k_shot, replace=False)
                unsampled = np.setdiff1d(cur_k_idxs, wg_ex)
                un_tup = (novel_class_idx, unsampled)
                # save unsampled to set against later
                cur_unsamp.append(un_tup)
                cur_subset = Subset(novel_ds, wg_ex)
                subset_dl = DataLoader(cur_subset, batch_size=k_shot, shuffle=False)
                #print(wg_k, unsampled)
                #print("for1")
                for ci,cl in subset_dl:
                    #print("cicl")
                    #print(f"before setting {cl}", model.classifier.cls_vec.shape)
                    #print(model.classifier.cls_vec)
                    #print("mapped_novel_idx", novel_ds.get_mapped_class_idx(novel_class_idx))
                    model.set_pseudonovel_vec(novel_ds.get_mapped_class_idx(novel_class_idx), ci.to(device))
                    #print(f"after setting {cl}", model.classifier.cls_vec[cl])
                    #print(model.classifier.cls_vec)
            novel_test_unsampled.append(cur_unsamp)
            #model.classifier.print_cls_vec_norms()
            for base_class_idx in base_class_idxs:
                cur_b_idxs = base_test_data.get_class_ex_idxs(base_class_idx)
                num_ex = len(cur_b_idxs)
                num_to_test = min(num_ex, max_num_test) if max_num_test > 0 else num_ex
                base_test_idxs = rng.choice(cur_b_idxs, size=num_to_test, replace=False)
                test_b = np.append(test_b, base_test_idxs)

            num_total = num_novel + num_base
            test_base_sb = Subset(base_test_data, test_b)
            test_base_dl = DataLoader(test_base_sb, batch_size=bs, shuffle=True)
            #print("model classifier shape", model.classifier.cls_vec.shape)
            # test base ~~~~~~~~~~~~~~~~~~~~~
            base_confmat_path = ""
            if to_print == True:
                print(f"--- Base Testing for Task {i} ---")
            #print(f"Before Base {model.classifier.cls_vec.requires_grad}")
            res_base = batch_handler(model, [test_base_dl], cur_loss, cur_opter=None, batch_type = batch_type, device=device, epoch_idx=epoch_idx, train_phase = train_phase, bs=bs, num_classes = num_total, to_print=to_print, to_time = to_time, modelname = modelname, dsname = baseset, ds_idx=epoch_idx, multilabel=multilabel)
            #print(f"After Base {model.classifier.cls_vec.requires_grad}")
            #res_base_arr.append(res_base)
            if to_res == True:
                UR.res_csv_appender(res_base, dest_dir=res_dir, expr_num = expr_num, epoch_idx=epoch_idx, batch_type=BatchType.test, baseset = baseset, novelset=novelset, modelname=modelname, train_phase=train_phase )
            if nep != None:
                UN.nep_batch_parser(nep, res_base,batch_type=batch_type, train_phase = train_phase, ds_type=DatasetType.base, modelname = modelname, dsname = baseset, ds_idx=epoch_idx)
            if to_graph == True:
                base_confmat_path = UR.plot_confmat(res_base['confmat'],multilabel=res_base['multilabel'],dest_dir=graph_dir, train_phase = train_phase, expr_num=expr_num, modelname = modelname, baseset=baseset, novelset=novelset, is_base = True)

            if len(base_confmat_path) > 0 and nep != None:
                UN.nep_confmat_upload(nep,base_confmat_path ,batch_type=BatchType.test, train_phase = train_phase, modelname = modelname, ds_type = DatasetType.base, dsname = baseset)

            # test every incremental novel dataset so far
            # doing this instantiation separately since want variety in examples tested against

            # accumulate novel dataloaders over tasks
            novel_dls = []
            novel_unsamp_classes = []
            for j,ds_unsamp_arr in enumerate(novel_test_unsampled):
                # for accumulated unsampled indices per ds
                ds_selected_unsampled = np.array([], dtype=int)
                for (unsamp_class, unsamp_idxs) in ds_unsamp_arr:
                    novel_unsamp_classes.append(unsamp_class)
                    num_unsampled = unsamp_idxs.shape[0]
                    #print(unsamp_class, num_unsampled, unsamp_idxs)
                    num_to_test = min(num_unsampled,max_num_test) if max_num_test > 0 else num_unsampled
                    unsampled_samp = rng.choice(unsamp_idxs, size=num_to_test, replace=False)
                    # add current class's unsampled indices to current ds's unsampled 
                    ds_selected_unsampled = np.append(ds_selected_unsampled, unsampled_samp)
                    # test against current novel dataset
                test_novel_sb = Subset(novel_test_datas[j][1], ds_selected_unsampled)
                test_novel_dl = DataLoader(test_novel_sb, batch_size=bs, shuffle=True)
                novel_dls.append(test_novel_dl)
            #print("novel_classes_test", novel_unsamp_classes)
            if to_print == True:
                print(f"--- Novel Testing for Task {i} ---")
            res_novel = batch_handler(model, novel_dls, cur_loss, cur_opter=None, batch_type = batch_type, device=device, epoch_idx=epoch_idx, train_phase = train_phase, bs=bs, num_classes = num_total, to_print=to_print, to_time = to_time, modelname=modelname, dsname=novelset, ds_idx=epoch_idx, multilabel=multilabel)

            novel_confmat_path = ""
            if nep != None:
                UN.nep_batch_parser(nep, res_novel,batch_type=batch_type, train_phase = train_phase, ds_type=DatasetType.novel, modelname = modelname, dsname = novelset, ds_idx=epoch_idx)
            if to_res == True:
                UR.res_csv_appender(res_novel, dest_dir=res_dir, expr_num = expr_num, epoch_idx=epoch_idx, batch_type=BatchType.test,modelname=modelname, baseset=baseset, novelset=novelset)
            if to_graph == True:
                novel_confmat_path = UR.plot_confmat(res_novel['confmat'],multilabel=res_novel['multilabel'],dest_dir=graph_dir, train_phase = train_phase, expr_num=expr_num, modelname = modelname, baseset=baseset, novelset=novelset, is_base = False)

            if len(novel_confmat_path) > 0 and nep != None:
                UN.nep_confmat_upload(nep,novel_confmat_path ,batch_type=BatchType.test, train_phase = train_phase, modelname = modelname, ds_type = DatasetType.novel, dsname = novelset)
            both_confmat_path = ''
            if len(novel_confmat_path) > 0 and  len(base_confmat_path) > 0:
                both_confmat_path = UR.plot_confmat(res_base['confmat'],confmat2=res_novel['confmat'],multilabel=res_novel['multilabel'],dest_dir=graph_dir, train_phase = train_phase, expr_num=expr_num, modelname = modelname, baseset=baseset, novelset=novelset, is_base = False)

            if len(both_confmat_path) > 0 and nep != None:
                UN.nep_confmat_upload(nep,both_confmat_path ,batch_type=BatchType.test, train_phase = train_phase, modelname = modelname, ds_type = DatasetType.novel, dsname = novelset, ds_idx=2)
        
        
    

    

def base_weightgen_trainer(model, cur_loss, cur_optim, train_data, valid_data, lr=1e-4, bs = 4, epochs = 1, save_ivl = 0, save_dir = UG.DEF_SAVEDIR, res_dir = UG.DEF_RESDIR, graph_dir = UG.DEF_GRAPHDIR, device = 'cpu', expr_num = 0, num_classes = 30, to_print = True, to_time = True, to_graph = True, to_res = True, rng = None, n_way = 5, k_shot = 5, total_novel_samp = 100, total_base_samp = 100, modelname = ModelName.samplecnn, baseset=DatasetName.esc50, novelset = DatasetName.esc50, multilabel=True, nep = None):
    model.set_train_phase(TrainPhase.base_weightgen)
    model.zero_grad()
    cur_optim.zero_grad()
    #unmapped base class indices
    base_classes = train_data.get_class_idxs()
    num_base_classes = len(base_classes)
    num_rest = num_base_classes - n_way
    # number of unsampled from novel to take for training against
    num_novel_unsampled = int((total_novel_samp - (n_way * k_shot))/n_way)
    # number of unsampled (all of them are) to take for training against (from rest of base classes)
    num_base_unsampled = int(total_base_samp/num_rest)
    res_wgen_batches = []
    res_valid_batches = []
    valid_dload = DataLoader(valid_data, shuffle=True, batch_size = bs)
    #print("called")
    if rng == None:
        rng = np.random.default_rng(seed=UG.DEF_SEED)
    for epoch_idx in range(epochs):
        #print("called")
        cur_novel = rng.choice(base_classes, size=n_way, replace=False)
        cur_base = np.setdiff1d(base_classes,cur_novel)
        # first sample from each class the examples to use to generate the classification vectors
        # and then some how sample other examples for t_novel and t_test (see gidaris supp)
        # and then do backprop to learn phi matrices and vectors
        # one idea is to sample t_novel from the remaining examples from the training set to there's only 8 vectors per class (wang doesn't specify where these come from)
        # another idea is to sample from validation set but then that might be information leakage
        # in any case, T_novel should be the same as t_base per class (ideally)
        # and we are only SIMULATING few-shot so it doesn't actually have to be fewshot
        # for learning novel classes, phi matrices should be fixed
       
        model.set_exclude_idxs(train_data.get_mapped_class_idxs(cur_novel), device=device)
        test_k = np.array([],dtype=int)
        test_b = np.array([], dtype=int)
        #print("for0")
        for novel_cls in cur_novel: # sampling for classification vectors
            cur_k_idxs = train_data.get_class_ex_idxs(novel_cls)
            # examples for weight generator
            wg_k = rng.choice(cur_k_idxs, size=k_shot, replace=False)
            unsampled = np.setdiff1d(cur_k_idxs, wg_k)
            cur_subset = Subset(train_data, wg_k)
            subset_dl = DataLoader(cur_subset, batch_size=k_shot, shuffle=False)
            #print(wg_k, unsampled)

            # pick from unsampled to test against
            unsampled_samp = rng.choice(unsampled, size=num_novel_unsampled, replace=False)
            test_k = np.append(test_k,unsampled_samp)
            #print("for1")
            for ci,cl in subset_dl:
                #print("cicl")
                model.set_pseudonovel_vec(train_data.get_mapped_class_idx(novel_cls), ci.to(device))
        for base_cls in cur_base:
            cur_b_idxs = train_data.get_class_ex_idxs(base_cls)
            base_test = rng.choice(cur_b_idxs, size=num_base_unsampled, replace=False)
            test_b = np.append(test_b, base_test)
        test_all = np.append(test_k,test_b)
        test_set = Subset(train_data, test_all)
        test_dl = DataLoader(test_set, batch_size=bs, shuffle=True) 
        model.weightgen_train_enable(True)
        if to_print == True:
            print(f"\nEpoch {epoch_idx}\n ==========================")
        res_wgen = batch_handler(model, [test_dl], cur_loss, cur_opter=cur_optim, batch_type = BatchType.train, device=device, epoch_idx=epoch_idx, train_phase = TrainPhase.base_weightgen, bs=bs, num_classes = num_classes, to_print=to_print, to_time = to_time, modelname=modelname, dsname = baseset, multilabel=multilabel)
        #print("got to here")
        if to_res == True:
            UR.res_csv_appender(res_wgen, dest_dir=res_dir, expr_num = expr_num, epoch_idx=epoch_idx, batch_type=BatchType.train, train_phase = TrainPhase.base_weightgen, baseset=baseset, novelset = novelset )

        if nep != None:
            UN.nep_batch_parser(nep, res_wgen,batch_type=BatchType.train, train_phase = TrainPhase.base_weightgen, ds_type = DatasetType.base, modelname=modelname, dsname=baseset, ds_idx=0)
        res_valid = batch_handler(model, [valid_dload], cur_loss, cur_opter=None, batch_type = BatchType.valid, device=device, epoch_idx=epoch_idx, bs=bs, to_print=to_print, to_time = to_time, num_classes = num_classes, modelname=modelname, dsname=baseset, multilabel=multilabel)

        if to_res == True:
            UR.res_csv_appender(res_valid, dest_dir=res_dir, expr_num = expr_num, epoch_idx=epoch_idx, batch_type=BatchType.valid, modelname=modelname, train_phase = TrainPhase.base_weightgen, baseset = baseset, novelset = novelset, pretrain = False)
        res_wgen_batches.append(res_wgen)
        res_valid_batches.append(res_valid)

        if nep != None:
            UN.nep_batch_parser(nep, res_valid,batch_type=BatchType.valid, train_phase = TrainPhase.base_weightgen, ds_type=DatasetType.base, modelname=modelname, dsname = baseset, ds_idx=0)

        # ~~~ end of loop save stuff ~~~
        model.weightgen_train_enable(False)
        model.reset_copies(copy_back=True,device=device)
        if save_ivl > 0:
            if ((epoch_idx +1) % save_ivl == 0 and epoch_idx != 0) or epoch_idx == (epochs-1):
                #model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_num=expr_num, modelname = modelname, baseset = baseset, novelset = novelset, model_idx = 1, mtype="embedder")
                model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_num=expr_num, modelname = modelname, baseset= baseset, novelset = novelset, model_idx=1, mtype="classifier")



    if to_graph == True:
        UR.train_valid_grapher(res_wgen_batches, res_valid_batches, dest_dir="graph", graph_key="loss_avg", expr_num=expr_num, baseset = baseset, novelset = novelset, modelname = modelname )
        UR.train_valid_grapher(res_valid_batches, res_valid_batches, dest_dir="graph", graph_key="time_avg", expr_num=expr_num,  baseset = baseset, novelset = novelset, modelname = modelname )


def base_init_trainer(model, cur_loss, cur_optim, train_data, valid_data, lr=1e-4, bs = 4, epochs = 1, save_ivl=0, save_dir=UG.DEF_SAVEDIR, res_dir = UG.DEF_RESDIR, graph_dir = UG.DEF_GRAPHDIR, device='cpu', expr_num = 0, num_classes = 30, to_print = True, to_time = True, to_graph = True, to_res = True, modelname = ModelName.samplecnn, baseset = DatasetName.esc50, novelset = DatasetName.esc50, multilabel=True, nep=None):
    model.set_train_phase(TrainPhase.base_init)
    train_dload = DataLoader(train_data, shuffle=True, batch_size = bs)
    valid_dload = DataLoader(valid_data, shuffle=True, batch_size = bs)
    #model.classifier.set_base_class_idxs(train_data.get_class_idxs())
    res_train_batches = []
    res_valid_batches = []
    for epoch_idx in range(epochs):
        if to_print == True:
            print(f"\nEpoch {epoch_idx}\n ==========================")
        res_train = batch_handler(model, [train_dload], cur_loss, cur_opter=cur_optim, batch_type = BatchType.train, device=device, train_phase=TrainPhase.base_init, epoch_idx=epoch_idx, bs=bs, num_classes= num_classes, to_print=to_print, to_time = to_time, modelname=modelname, dsname=baseset, multilabel=multilabel)
        if to_res == True:
            UR.res_csv_appender(res_train, dest_dir=res_dir, expr_num = expr_num, epoch_idx=epoch_idx, batch_type=BatchType.train, modelname = modelname, baseset = baseset, novelset = novelset)
        if save_ivl > 0:
            if ((epoch_idx +1) % save_ivl == 0 and epoch_idx != 0) or epoch_idx == (epochs-1):
                model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_num=expr_num, modelname=modelname, baseset = baseset, novelset = novelset, model_idx = 0, mtype="embedder")
                model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_num=expr_num, modelname=modelname, baseset = baseset, novelset = novelset, model_idx = 0, mtype="classifier")

        if nep != None:
            UN.nep_batch_parser(nep, res_train,batch_type=BatchType.train, train_phase = TrainPhase.base_init, ds_type = DatasetType.base, modelname=modelname, dsname=baseset, ds_idx=0)
        res_valid = batch_handler(model, [valid_dload], cur_loss, cur_opter=None, batch_type = BatchType.valid, device=device, train_phase=TrainPhase.base_init, epoch_idx=epoch_idx, bs=bs, to_print=to_print, to_time = to_time, multilabel=multilabel, modelname=modelname, dsname=baseset, num_classes = num_classes)
        if to_res == True:
            UR.res_csv_appender(res_valid, dest_dir=res_dir, expr_num = expr_num, epoch_idx=epoch_idx, batch_type=BatchType.valid, train_phase =TrainPhase.base_init, modelname=modelname, baseset = baseset, novelset = novelset, pretrain = False)
        res_train_batches.append(res_train)
        res_valid_batches.append(res_valid)

        if nep != None:
            UN.nep_batch_parser(nep, res_valid,batch_type=BatchType.valid, train_phase = TrainPhase.base_init, ds_type=DatasetType.base, modelname=modelname, dsname=baseset, ds_idx=0)
    if to_graph == True:
        UR.train_valid_grapher(res_train_batches, res_valid_batches, dest_dir="graph", graph_key="loss_avg", expr_num=expr_num, modelname = modelname, baseset = baseset, novelset = novelset )
        UR.train_valid_grapher(res_train_batches, res_valid_batches, dest_dir="graph", graph_key="time_avg", expr_num=expr_num, modelname = modelname, baseset = baseset, novelset = novelset )


if __name__ == "__main__":
    expr_num = int(time.time() * 1000)
    args = UP.parse_args()
    #print(args)
    make_folder(args["save_ivl"] > 0, args["save_dir"]) 
    make_folder(args["to_graph"], args["graph_dir"]) 
    make_folder(args["to_res"], args["res_dir"]) 
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
    #rese2_list_old = [(2,3,128,3), (7,3,256,3),(1,3,512,3)]

    #to make compatible with cnn14
    rese2_list = [(2,3,128,3), (2,3,256,3),  (2,3,512,3),  (3,3,1024,3), (1,3,2048,3)]
    #se_list = [(2,3,128,3), (7,3,256,3),(1,2,512,2)]
    se_list = []
    res2_list = []
    simple_list = []
    # middle dim according to (1) is same as num channels
    num_classes_base = 30
    num_classes_valid = 10
    num_classes_test = 10
    device = 'cpu'
    t_ph = TrainPhase[args["train_phase"]]
    #max_samp = 118098
    max_samp = 177147

    if torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()

    model = None
    mtype = ModelName[args["model"]]
    if args["model"] == "samplecnn":
        model = SampCNNModel(in_ch=1, strided_list=strided_list, basic_list=[], res1_list=res1_list, res2_list=res2_list, se_list=se_list, rese1_list=[], rese2_list=rese2_list, simple_list=simple_list, se_dropout=args["se_dropout"], res1_dropout=args["res1_dropout"], res2_dropout=args["res2_dropout"], rese1_dropout=args["rese1_dropout"], rese2_dropout=args["rese2_dropout"],simple_dropout=args["simple_dropout"], se_fc_alpha=args["se_fc_alpha"], rese1_fc_alpha=args["rese1_fc_alpha"], rese2_fc_alpha=args["rese2_fc_alpha"], num_classes_base=num_classes_base, num_classes_novel = 0, sr=args["sample_rate"], omit_last_relu = args["omit_last_relu"], use_prelu = args["use_prelu"], se_prelu=args["se_prelu"], cls_fn = args["cls_fn"]).to(device)
    else:
        
        mtype = ModelName.cnn14
        model = CNN14Model(in_ch=1, num_classes_base=num_classes_base, num_classes_novel=0, sr=args["sample_rate"], dropout = args["dropout"], seed=3, omit_last_relu = args["omit_last_relu"], train_phase = t_ph, use_prelu = args["use_prelu"], use_bias = args["use_bias"], cls_fn = args["cls_fn"]).to(device)

    #save_str=f"{expr_num}-{modelname.name}-{dsname.name}_{mtype}_{epoch_idx}-model.pth"
    modelname = args["model"]
    baseset = args["baseset"]
    load_emb = args["load_emb"]
    load_cls = args["load_cls"]
    emb_idx = args["emb_idx"]
    cls_idx = args["cls_idx"]
    if args["expr_num"] >= 0 and args["emb_load_num"] >= 0 and args["cls_load_num"] >= 0:
        expr_num = args["expr_num"]
        emb_load_num = args["emb_load_num"]
        cls_load_num = args["cls_load_num"]
        load_emb_fname = f"{expr_num}_{emb_idx}-{modelname}-{baseset}_embedder_{emb_load_num}-model.pth"
        load_cls_fname = f"{expr_num}_{cls_idx}-{modelname}-{baseset}_classifier_{cls_load_num}-model.pth"
        load_emb = os.path.join(args["model_dir"], load_emb_fname)
        load_cls = os.path.join(args["model_dir"], load_cls_fname)
    if ".pth" in load_emb:
        load_file = load_emb
        if args["expr_num"] > 0 == False:
            expr_num = int(load_file.split(os.sep)[-1].split("-")[0])
        #t_ph = TrainPhase.base_weightgen
        model.embedder.load_state_dict(torch.load(load_emb))
        print(f"loaded {load_emb}")

    if ".pth" in load_cls:
        model.classifier.load_state_dict(torch.load(load_cls))
        print(f"loaded {load_cls}")


    nrun = None 
    # NEPTUNE STUFF
    if args["to_nep"] == True:
        nep_api = ""
        print(f"running neptune: {args['to_nep']}")
        with open(UG.DEF_NEP_API, "r") as f:
            nep_api = f.read().strip()


        nrun = neptune.init_run(
            project="Soundbendor/dfsl-rawwv",
            api_token= nep_api,
            capture_hardware_metrics=False,
            )

        nrun["model/params"] = args

    #print(model.embedder.state_dict())
    bstype = DatasetName[args["baseset"]]
    nstype = DatasetName[args["novelset"]]
    runner(model, train_phase = t_ph,expr_num = expr_num, lr=args["learning_rate"], bs=args["batch_size"],
            base_epochs=args["base_epochs"], weightgen_epochs = args["weightgen_epochs"], novel_epochs = args["novel_epochs"],
            save_ivl = args["save_ivl"], sr = args["sample_rate"], max_samp = max_samp, use_class_weights = args["use_class_weights"], label_smoothing = args["label_smoothing"], 
            multilabel=args["multilabel"], res_dir=args["res_dir"], save_dir=args["save_dir"],
            to_print=args["to_print"], to_time=args["to_time"], graph_dir = args["graph_dir"], data_dir=args["data_dir"], to_graph=args["to_graph"], to_res=args["to_res"],
            device=device, nep = nrun, baseset = bstype, novelset = nstype,
            n_way = args["n_way"], k_shot = args["k_shot"], modelname = mtype
            )
    if args["to_nep"] == True:
        nrun.stop()
