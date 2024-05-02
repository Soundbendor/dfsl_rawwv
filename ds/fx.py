import torch
import torchaudio.functional as TAF
from torch import nn
import ds.sndload as SND
import os
curpath = os.getcwd()
fxpath = os.path.join(curpath, '..', 'fx')
irpath = os.path.join(fxpath,'252847__kijjaz__20141025-kijjaz-ir-new-office-01.aiff')
bnoizpath = os.path.join(fxpath, 'bnoiz5s.wav')
wnoizpath = os.path.join(fxpath, 'wnoiz5s.wav')
pnoizpath = os.path.join(fxpath, 'pnoiz5s.wav')


ipls_resp =SND.sndloader(irpath, want_sr = 16000, to_mono=True)

# noise generated through sox -n filename.wav synth 5 white/pink/brownnoise
bnoiz = SND.sndloader(bnoizpath, want_sr = 16000, to_mono=True)
wnoiz = SND.sndloader(wnoizpath, want_sr = 16000, to_mono=True)
pnoiz = SND.sndloader(pnoizpath, want_sr = 16000, to_mono=True)

def apply_reverb(ipt_snds):
    aug = TAF.fftconvolve(ipt_snds, ipls_resp, mode='same')
    return aug

def apply_lpf(ipt_snds, freq=500, q = 0.707):
    aug = TAF.lowpass_biquad(ipt_snds, 16000, cutoff_freq=freq), Q=q)
    return aug

def apply_hpf(ipt_snds, freq=500, q = 0.707):
    aug = TAF.highpass_biquad(ipt_snds, 16000, cutoff_freq=freq), Q=q)
    return aug

def apply_bpf(ipt_snds, freq=500, q = 0.707):
    aug = TAF.bandpass_biquad(ipt_snds, 16000, central_freq=freq), Q=q)
    return aug

def apply_noise(ipt_snds,ntype='white',snr=10):
    cur_noiz = wnoiz
    if ntype == 'pink':
        cur_noiz = pnoiz
    elif ntype == 'brown':
        cur_noiz = bnoiz

    aug = TAF.add_noise(ipt_snds, cur_noiz, torch.tensor(snr))
    return aug
