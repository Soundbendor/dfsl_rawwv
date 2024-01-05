import librosa as LR
import numpy as np
import torch
import torchaudio as TA
import torchaudio.functional as TAF
import torchaudio.transforms as TAT
import matplotlib.pyplot as plt
import os
import sys

def sndloader(want_file:str, want_sr:int = None, want_bits:int = None, max_samp = np.inf, to_mono:bool = True) -> any:
    read_frames = int(max_samp) if np.isfinite(max_samp) == True else -1
    cur_wf, cur_sr = TA.load(want_file, num_frames = read_frames)
    cur_nc, cur_nf = cur_wf.shape
    ret_wf = None
    ret_sr = None

    cur_wf2 = None
    if to_mono == True and cur_nc > 1:
        cur_mult = 1./float(cur_nc)
        cur_wf2 = (cur_wf * cur_mult).sum(dim=0).unsqueeze(0)
    else:
        cur_wf2 = cur_wf
    if want_sr != cur_sr and want_sr != None:
        ret_wf = resamp_helper(cur_wf2, cur_sr, want_sr, want_bits=want_bits)
        ret_sr = want_sr
    elif want_bits != None:
        cur_dtype = cur_wf.dtype
        cur_bits = 16
        if cur_dtype == torch.float64:
            cur_bits = 64
        elif cur_dtype == torch.float32:
            cur_bits = 32
        elif cur_dtype == torch.float16:
            cur_bits = 16
        if want_bits != cur_bits:

            want_dtype = None
            if want_bits != None:
                if want_bits == 64:
                    want_dtype = torch.float64
                elif want_bits == 32:
                    want_dtype = torch.float32
                elif want_bits == 16:
                    want_dtype = torch.float16
                else:
                    want_dtype = torch.float16
            ret_wf = cur_wf2.to(want_dtype)
            ret_sr = cur_sr
    else:
        ret_wf = cur_wf2
        ret_sr = cur_sr

    if ret_wf.shape[1] < read_frames:
        ret_wf = torch.nn.functional.pad(ret_wf, (0, read_frames - ret_wf.shape[1]))

    return ret_wf

def resamp_helper(cur_wf:any, cur_sr:int, want_sr:int, want_bits:int = None) -> any:
    cur_dtype = cur_wf.dtype
    resamp_f = TAT.Resample(cur_sr, want_sr)
    want_dtype = None
    resamped = None
    if want_bits != None:
        if want_bits == 64:
            want_dtype = torch.float64
        elif want_bits == 32:
            want_dtype = torch.float32
        elif want_bits == 16:
            want_dtype = torch.float16
    if want_dtype == None:
        resamped = resamp_f(cur_wf)
    else:
        resamped = resamp_f(cur_wf).to(want_dtype)
    return resamped

# https://pytorch.org/audio/stable/tutorials/mvdr_tutorial.html#sphx-glr-tutorials-mvdr-tutorial-py
def plot_spectrogram(stft, title="Spectrogram"):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(spectrogram, cmap="viridis", vmin=-100, vmax=0, origin="lower", aspect="auto")
    axis.set_title(title)
    plt.colorbar(img, ax=axis)
 
