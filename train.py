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

#REFERENCES:
# (1) Kim, T. (2019) sampleaudio [Github Repository]. Github. https://github.com/tae-jun/sampleaudio/
# (2) Lee, J., Park, J., Kim, K. L, and Nam, J. (2018). SampleCNN: End-to-End Deep Convolutional Neural Networks Using Very Small Filters for Music Classification. Applied Sciences 8(1). https://doi.org/10.3390/app8010150
# (3) Kim, T., Lee, J., and Nam, J. (2019). Comparison and Analysis of SampleCNN Architectures for Audio Classification. IEEE Journal of Selected Topics in Signal Processing 13(2), 285-297. https://doi.org/10.1109/JSTSP.2019.2909479
# (4) Wang, Y., Bryan, N. J., Cartwright, M., Bello, J. P., and Salamon, J. (2021a). Few-Shot Continual Learning for Audio Classification. ICASSP 2021 - 2021 IEEE International Conference on Acoustic, Speech and Signal Processing, 321-325. https://doi.org/10.1109/ICASSP39728.2021.9413584.

# ESC50: 30 (base)-10 novel(val) - 10 novel(test) class split
# 24 (training):8(validation):8(testing) sample split

DEF_DATADIR = os.path.join(os.sep, 'media', 'dxk', 'tosh_ext', 'ds', 'ESC-50-master') 
DEF_SAVEDIR = os.path.join(os.path.split(__file__)[0], "save")

def runner(model, to_train = True, seed=3, max_rng=10000, lr = 1e-4, bs=4, save_dir = DEF_SAVEDIR, data_dir = DEF_DATADIR, epochs=1, save_ivl=0, to_print=True, device='cpu'):
    _sr = 44100
    _max_samp = 236196
    rng = np.random.default_rng(seed=seed)
    cur_seed = rng.integers(0,max_rng,1)[0]
    torch.manual_seed(seed)
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

    trainer(model,base_train_data,base_valid_data,lr=lr, bs=bs, save_ivl=save_ivl, save_dir=save_dir, to_print=to_print, device=device)

def loss_printer(epoch_idx, batch_idx, cur_loss, loss_type='train', to_print = True):
    if to_print == True:
        cur_str = f"{loss_type} loss ({epoch_idx},{batch_idx}): {cur_loss}"
        print(cur_str)

def trainer_batch(model, epoch_idx, dloader, cur_losser, cur_opter, device='cpu', to_print = True):
    for batch_idx, (ci,cl) in enumerate(dloader):
        model.zero_grad()
        pred = model(ci.to(device))
        #print(ci.shape)
        #print(cl.shape)
        cur_loss = cur_losser(pred, cl.type(torch.FloatTensor).to(device))
        cur_loss.backward()
        cur_opter.step()
        loss_item = cur_loss.item()
        loss_printer(epoch_idx, batch_idx, loss_item, loss_type='train', to_print = to_print)


def valid_batch(model, epoch_idx, dloader, cur_losser, device='cpu', to_print = True):
    model.eval()
    for batch_idx, (ci,cl) in enumerate(dloader):
        with torch.no_grad():
            pred = model(ci.to(device))
            cur_loss = cur_losser(pred, cl.type(torch.FloatTensor).to(device))
            #print(ci.shape)
            #print(cl.shape)
            loss_item = cur_loss.item()
            loss_printer(epoch_idx, batch_idx, loss_item, loss_type='valid', to_print = to_print)


def model_saver(cur_model, save_dir=DEF_SAVEDIR, epoch_idx=0, mtype="embedder"):
    ctime = int(time.time() * 1000)
    save_str=f"sampcnn_{mtype}-{epoch_idx}_{ctime}.pth"
    outpath = os.path.join(save_dir, save_str)
    cdict = None
    if mtype=="embedder":
        cdict = cur_model.embedder.state_dict()
    else:
        cdict = cur_model.classifier.state_dict()
    torch.save(cdict, outpath)



def trainer(model, train_data, valid_data, lr=1e-4, bs = 4, epochs = 1, save_ivl=0, save_dir=DEF_SAVEDIR, device='cpu', to_print = True):
    train_dload = DataLoader(train_data, shuffle=True, batch_size = bs)
    valid_dload = DataLoader(valid_data, shuffle=True, batch_size = bs)
    cur_loss = nn.CrossEntropyLoss()
    cur_optim = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch_idx in range(epochs):
        if to_print == True:
            print(f"Epoch {epoch_idx}\n ==========================")
        trainer_batch(model, epoch_idx, train_dload, cur_loss, cur_optim, to_print=to_print, device=device)
        if save_ivl > 0:
            if epoch_idx % save_ivl == 0 or epoch_idx == (epochs-1):
                model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, mtype="embedder")
                #model_saver(model, save_dir=save_dir, epoch_idx=epoch_idx, mtype="classifier")
        valid_batch(model, epoch_idx, valid_dload, cur_loss, to_print=to_print, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--bs", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=1, help="batch size")
    parser.add_argument("--save_ivl", type=int, default=0, help="(epoch interval) to save model (<= 0: don't save)")
    parser.add_argument("--save_dir", type=str, default=DEF_SAVEDIR, help="save directory")
    parser.add_argument("--data_dir", type=str, default=DEF_DATADIR, help="root of data directory")
    parser.add_argument("--load_emb", type=str, default='', help="load embedder with given pth file")
    parser.add_argument("--load_cls", type=str, default='', help="load classifier with given pth file")
    parser.add_argument("--to_print", type=bool, default=True, help="print progress")
    args = parser.parse_args()
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

    if torch.cuda.is_available() == True:
        device = 'cuda'

    model = SampCNNModel(in_ch=1, strided_list=strided_list, basic_list=[], res1_list=res1_list, res2_list=[], se_list=[], rese1_list=[], rese2_list=[], simple_list=simple_list, res1_dropout=0.2, res2_dropout=0.2, rese1_dropout=0.2, rese2_dropout=0.2,simple_dropout=0.5, se_fc_alpha=2.**(-3), rese1_fc_alpha=2.**(-3), rese2_fc_alpha=2.**(-3), use_classifier=True,fc_dim=fc_dim, num_classes=num_classes, sr=44100).to(device)
    if ".pth" in args.load_emb:
        model.embedder.load_state_dict(torch.load(args.load_emb))

    if ".pth" in args.load_cls:
        model.classifier.load_state_dict(torch.load(args.load_cls))
    #print(model.embedder.state_dict())
    runner(model, lr=args.lr, bs=args.bs, epochs=args.epochs, save_ivl = args.save_ivl,
            data_dir=args.data_dir, save_dir=args.save_dir, to_print=args.to_print, device=device)
