import numpy as np
import torch
import torchaudio as TA
from torch import nn
from torch.utils.data import DataLoader
from torch import cuda
from arch.sampcnn_model import SampCNNModel
from ds.esc50 import ESC50 
import os
import argparse
import time
import contextlib
from util.types import BatchType
import util.results as UR 
#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479
# (4) Wang, Y., Bryan, N. J., Cartwright, M., Bello, J. P., and Salamon, J. (2021a). Few-Shot Continual Learning for Audio Classification. ICASSP 2021 - 2021 IEEE International Conference on Acoustic, Speech and Signal Processing, 321-325. https://doi.org/10.1109/ICASSP39728.2021.9413584.

# ESC50: 30 (base)-10 novel(val) - 10 novel(test) class split
# 24 (training):8(validation):8(testing) sample split

DEF_DATADIR = os.path.join(os.sep, 'media', 'dxk', 'tosh_ext', 'ds', 'ESC-50-master') 
DEF_SAVEDIR = os.path.join(os.sep, 'media', 'dxk', 'tosh_ext', 'fscil', 'dfsl_rawwv', 'save') 
#DEF_SAVEDIR = os.path.join(os.path.split(__file__)[0], "save")
DEF_GRAPHDIR = os.path.join(os.path.split(__file__)[0], "graph")
DEF_RESDIR = os.path.join(os.path.split(__file__)[0], "res")

def make_folder(cur_arg, cur_dir):
    if os.path.exists(cur_dir) != True and cur_arg == True:
        os.makedirs(cur_dir)

def runner(model, expr_idx = 0, to_train = True, seed=3, max_rng=10000, lr = 1e-4, bs=4, save_dir = DEF_SAVEDIR, res_dir = DEF_RESDIR, data_dir = DEF_DATADIR, epochs=1, save_ivl=0, to_print=True, to_time = True, to_graph=True, to_res = True, device='cpu'):
    _sr = 44100
    _max_samp = 236196
    rng = np.random.default_rng(seed=seed)
    cur_seed = rng.integers(0,max_rng,1)[0]
    torch.manual_seed(seed)
    cur_loss = nn.CrossEntropyLoss()

    class_order = np.arange(0,50) # order of classes
    rng.shuffle(class_order) # shuffle classes
    base_classes = class_order[:30]
    novelval_classes = class_order[30:40]
    noveltest_classes = class_order[40:]
    fold_order = np.arange(1,6) # order of folds
    rng.shuffle(fold_order) # shuffle folds to group folds other than sequentially
    training_folds = fold_order[:3]
    valid_folds = fold_order[3:4]
    test_folds = fold_order[4:]
    base_train_data = ESC50(folds=training_folds, classes=base_classes, k=24, srate=_sr, samp_sz=_max_samp, basefolder = data_dir, seed = cur_seed)
    base_valid_data = ESC50(folds=valid_folds, classes=base_classes, k=24, srate=_sr, samp_sz=_max_samp, basefolder = data_dir, seed = cur_seed)
    base_test_data = ESC50(folds=test_folds, classes=base_classes, k=24, srate=_sr, samp_sz=_max_samp, basefolder = data_dir, seed = cur_seed)
    novelval_train_data = ESC50(folds=training_folds, classes=novelval_classes, k=8, srate=_sr, samp_sz=_max_samp, basefolder = data_dir, seed = cur_seed)
    novelval_valid_data = ESC50(folds=valid_folds, classes=novelval_classes, k=8, srate=_sr, samp_sz=_max_samp, basefolder = data_dir, seed = cur_seed)
    novelval_test_data = ESC50(folds=test_folds, classes=novelval_classes, k=8, srate=_sr, samp_sz=_max_samp, basefolder = data_dir, seed = cur_seed)
    noveltest_train_data = ESC50(folds=training_folds, classes=noveltest_classes, k=8, srate=_sr, samp_sz=_max_samp, basefolder = data_dir, seed = cur_seed)
    noveltest_valid_data = ESC50(folds=valid_folds, classes=noveltest_classes, k=8, srate=_sr, samp_sz=_max_samp, basefolder = data_dir, seed = cur_seed)
    noveltest_test_data = ESC50(folds=test_folds, classes=noveltest_classes, k=8, srate=_sr, samp_sz=_max_samp, basefolder = data_dir, seed = cur_seed)
    print("~~~~~")
    print(f"Running Expr {expr_idx} with epochs: {epochs}, bs:{bs}, lr:{lr}\n-----")
    print(f"Training: {to_train}, Printing: {to_print}, Save Model Interval: {save_ivl}, Graphing: {to_graph}, Saving Results: {to_res}")
    print("~~~~~")
    if to_train == True:
        trainer(model,cur_loss,base_train_data,base_valid_data, expr_idx= expr_idx, epochs=epochs, lr=lr, bs=bs, save_ivl=save_ivl, save_dir=save_dir, to_print=to_print, to_time=to_time, to_graph=to_graph, to_res=to_res,device=device)
    tester(model,cur_loss,base_test_data, expr_idx= expr_idx, bs=bs, to_print=to_print, to_time=to_time, to_graph=to_graph, to_res=to_res,device=device,pretrain=(to_train == False))

def loss_printer(epoch_idx, batch_idx, cur_loss, loss_type=BatchType.train, to_print = True):
    if to_print == True:
        cur_str = f"{loss_type.name} loss ({epoch_idx},{batch_idx}): {cur_loss}"
        print(cur_str)


def batch_handler(model, dloader, cur_losser, cur_opter=None, batch_type = BatchType.train, device='cpu', bs=4, epoch_idx=0, to_print=True, to_time = False):
    #time_batch = []
    loss_batch = []
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
            cur_loss = cur_losser(pred, cl.type(torch.FloatTensor).to(device))
            loss_item = cur_loss.item()
            if train ==True:
                cur_loss.backward()
                cur_opter.step()
                cur_opter.zero_grad()
            if to_print == True:
                loss_printer(epoch_idx, batch_idx, loss_item, loss_type=batch_type, to_print = to_print )
            loss_batch.append(loss_item)
            """
            if to_time == True:
                time_finish = time.time()
                batch_times.append(time_finish - time_last)
                time_last = time_start
            """
    time_avg = -1
    time_batch_overall = -1
    if to_time == True:
        #time_avg = np.mean(time_batch)
        time_batch_overall = time.time() - time_start
        time_avg = time_batch_overall/bs
    loss_avg = np.mean(loss_batch)
    ret = {"epoch_idx": epoch_idx, "batch_type": batch_type.name,
            "batch_avg_loss": loss_avg, "batch_avg_time": time_avg}
    return ret

def model_saver(cur_model, save_dir=DEF_SAVEDIR, epoch_idx=0, expr_idx = 0, mtype="embedder"):
    save_str=f"{expr_idx}-sampcnn_{mtype}_{epoch_idx}-model.pth"
    outpath = os.path.join(save_dir, save_str)
    cdict = None
    if mtype=="embedder":
        cdict = cur_model.embedder.state_dict()
    else:
        cdict = cur_model.classifier.state_dict()
    torch.save(cdict, outpath)


def tester(model, cur_loss, test_data, bs = 4, res_dir = DEF_RESDIR, device='cpu', expr_idx = 0, to_print = True, to_time = True, to_graph = True, to_res = True, pretrain = False):
    test_dload = DataLoader(test_data, shuffle=True, batch_size = bs)
    if to_print == True:
        print(f"\n Testing\n ==========================")
    res_test = batch_handler(model, test_dload, cur_loss, cur_opter=None, batch_type = BatchType.test, device=device, epoch_idx=-1, bs=bs, to_print=to_print, to_time = to_time)
    if to_res == True:
        UR.res_csv_appender(res_test, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=-1, batch_type=BatchType.test, expr_name="sampcnn_base", pretrain=pretrain)






def trainer(model, cur_loss, train_data, valid_data, lr=1e-4, bs = 4, epochs = 1, save_ivl=0, save_dir=DEF_SAVEDIR, res_dir = DEF_RESDIR, graph_dir = DEF_GRAPHDIR, device='cpu', expr_idx = 0, to_print = True, to_time = True, to_graph = True, to_res = True):
    train_dload = DataLoader(train_data, shuffle=True, batch_size = bs)
    valid_dload = DataLoader(valid_data, shuffle=True, batch_size = bs)
    cur_optim = torch.optim.Adam(model.parameters(), lr=lr)
    res_train_batches = []
    res_valid_batches = []
    if to_res == True:
        settings_dict = {"lr": lr, "bs": bs, "epochs": epochs}
        UR.settings_csv_writer(settings_dict, dest_dir = res_dir, expr_idx = expr_idx, expr_name="sampcnn_base")
    for epoch_idx in range(epochs):
        if to_print == True:
            print(f"\nEpoch {epoch_idx}\n ==========================")
        res_train = batch_handler(model, train_dload, cur_loss, cur_opter=cur_optim, batch_type = BatchType.train, device=device, epoch_idx=epoch_idx, bs=bs, to_print=to_print, to_time = to_time)
        if to_res == True:
            UR.res_csv_appender(res_train, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=epoch_idx, batch_type=BatchType.train, expr_name="sampcnn_base")
        if save_ivl > 0:
            if ((epoch_idx +1) % save_ivl == 0 and epoch_idx != 0) or epoch_idx == (epochs-1):
                model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, expr_idx=expr_idx, mtype="embedder")
                #model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, mtype="classifier")


        res_valid = batch_handler(model, valid_dload, cur_loss, cur_opter=None, batch_type = BatchType.valid, device=device, epoch_idx=epoch_idx, bs=bs, to_print=to_print, to_time = to_time)
        if to_res == True:
            UR.res_csv_appender(res_valid, dest_dir=res_dir, expr_idx = expr_idx, epoch_idx=epoch_idx, batch_type=BatchType.valid, expr_name="sampcnn_base", pretrain = False)
        res_train_batches.append(res_train)
        res_valid_batches.append(res_valid)
    if to_graph == True:
        UR.train_valid_loss_grapher(res_train_batches, res_valid_batches, dest_dir="graph", expr_idx=expr_idx, expr_name="sampcnn_base", pretrain = False)


if __name__ == "__main__":
    expr_idx = int(time.time() * 1000)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lr", type=float, default=1e-6, help="learning rate")
    parser.add_argument("--bs", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="batch size")
    parser.add_argument("--save_ivl", type=int, default=0, help="(epoch interval) to save model (<= 0: don't save)")
    parser.add_argument("--data_dir", type=str, default=DEF_DATADIR, help="base folder of dataset")
    parser.add_argument("--save_dir", type=str, default=DEF_SAVEDIR, help="save directory")
    parser.add_argument("--res_dir", type=str, default=DEF_RESDIR, help="results (textual) directory")
    parser.add_argument("--graph_dir", type=str, default=DEF_GRAPHDIR, help="graph directory")
    parser.add_argument("--load_emb", type=str, default='', help="load embedder with given pth file")
    parser.add_argument("--load_cls", type=str, default='', help="load classifier with given pth file")
    parser.add_argument("--to_print", type=bool, default=True, help="print progress")
    parser.add_argument("--to_time", type=bool, default=True, help="time inference/back prop")
    parser.add_argument("--to_graph", type=bool, default=True, help="save graphs")
    parser.add_argument("--to_res", type=bool, default=True, help="save result (textual) data")
    args = parser.parse_args()
    #print(args)
    make_folder(args.save_ivl > 0, args.save_dir) 
    make_folder(args.to_graph, args.graph_dir) 
    make_folder(args.to_res, args.res_dir) 
    #(num,ksize,stride) = (10,3,3),(2,2,2) gives 236196 which is 15696 extra samples 
    # (includes starting strided conv and following regular conv with strided maxpool)
    # but doesn't include 1 channel conv
    
    #(1) multiplies channels by 2 if 3rd block after strided or if last "config block"
    # also omits stride 1 conv as found in (3)

    # (num ksize, out_ch, stride)
    strided_list = [(1,3,128,3)]
    res1_list = [(2,3,128,3), (7,3,256,3),(1,2,256,2), (1,2,512,2)]
    simple_list = []
    # middle dim according to (1) is same as num channels
    fc_dim = 512
    num_classes = 50
    device = 'cpu'
    to_train = True

    if torch.cuda.is_available() == True:
        device = 'cuda'

    model = SampCNNModel(in_ch=1, strided_list=strided_list, basic_list=[], res1_list=res1_list, res2_list=[], se_list=[], rese1_list=[], rese2_list=[], simple_list=simple_list, res1_dropout=0.2, res2_dropout=0.2, rese1_dropout=0.2, rese2_dropout=0.2,simple_dropout=0.5, se_fc_alpha=2.**(-3), rese1_fc_alpha=2.**(-3), rese2_fc_alpha=2.**(-3), use_classifier=True,fc_dim=fc_dim, num_classes=num_classes, sr=44100).to(device)
    if ".pth" in args.load_emb:
        load_file = args.load_emb
        expr_idx = int(load_file.split(os.sep)[-1].split("-")[0])
        to_train = False
        model.embedder.load_state_dict(torch.load(args.load_emb))

    if ".pth" in args.load_cls:
        model.classifier.load_state_dict(torch.load(args.load_cls))
    #print(model.embedder.state_dict())
    runner(model, to_train=to_train,expr_idx = expr_idx, lr=args.lr, bs=args.bs, epochs=args.epochs, save_ivl = args.save_ivl,
            res_dir=args.res_dir, save_dir=args.save_dir, to_print=args.to_print, to_time=args.to_time, data_dir=args.data_dir, to_graph=args.to_graph, to_res=args.to_res, device=device)
