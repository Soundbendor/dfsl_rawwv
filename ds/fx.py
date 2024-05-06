import torch
import torchaudio.functional as TAF
from torch import nn
import ds.sndload as SND
import os,sys
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
from util.types import AugType

augtypes = [x.name for x in AugType if (x.name != 'random')]
num_augtypes = len(augtypes)

curpath = os.path.split(__file__)[0]
fxpath = os.path.join(curpath , '..', 'fx')
irpath = os.path.join(fxpath,'252847__kijjaz__20141025-kijjaz-ir-new-office-01.aiff')
bnoizpath = os.path.join(fxpath, 'bnoiz20s.wav')
wnoizpath = os.path.join(fxpath, 'wnoiz20s.wav')
pnoizpath = os.path.join(fxpath, 'pnoiz20s.wav')


# noise generated through sox -r 44100 -n filename.wav synth 20 white/pink/brownnoise
ipls_resp = None
bnoiz = None
wnoiz = None
pnoiz = None
def load_sounds(device='cpu'): 
    global ipls_resp,bnoiz,wnoiz,pnoiz
    ipls_resp =SND.sndloader(irpath, want_sr = 16000, to_mono=True).unsqueeze(0).to(device)
    bnoiz = SND.sndloader(bnoizpath, want_sr = 16000, to_mono=True).unsqueeze(0).to(device)
    wnoiz = SND.sndloader(wnoizpath, want_sr = 16000, to_mono=True).unsqueeze(0).to(device)
    pnoiz = SND.sndloader(pnoizpath, want_sr = 16000, to_mono=True).unsqueeze(0).to(device)

#print(bnoiz.shape, wnoiz.shape, pnoiz.shape)
def apply_reverb(ipt_snds):
    n = ipt_snds.shape[0]
    aug = TAF.fftconvolve(ipt_snds, ipls_resp.repeat(n,1,1), mode='same')
    return aug

def apply_lpf(ipt_snds, freq=500, q = 0.707):
    aug = TAF.lowpass_biquad(ipt_snds, 16000, cutoff_freq=freq, Q=q)
    return aug

def apply_hpf(ipt_snds, freq=500, q = 0.707):
    aug = TAF.highpass_biquad(ipt_snds, 16000, cutoff_freq=freq, Q=q)
    return aug

def apply_bpf(ipt_snds, freq=500, q = 0.707):
    aug = TAF.bandpass_biquad(ipt_snds, 16000, central_freq=freq, Q=q)
    return aug

def apply_noise(ipt_snds,ntype='white',snr=10):
    cur_noiz = wnoiz
    if ntype == 'pink':
        cur_noiz = pnoiz
    elif ntype == 'brown':
        cur_noiz = bnoiz

    n = ipt_snds.shape[0]
    t = ipt_snds.shape[-1]
    #print(t, ipt_snds.shape, cur_noiz[:,:,:t].repeat(n,1,1).shape)
    aug = TAF.add_noise(ipt_snds, cur_noiz[:,:,:t].repeat(n,1,1), torch.tensor(snr).unsqueeze(0).repeat(n,1))
    return aug


def apply_pshift(ipt_snds, n_steps=0):
    ret = TAF.pitch_shift(ipt_snds, 16000, n_steps, bins_per_octave=12, n_fft = 1024)
    return ret

def apply_random_fx(ipt_snds, rng):
    fxtype_idx = rng.integers(0,num_augtypes)
    fxtype = AugType[augtypes[fxtype_idx]]
    ret = apply_fx(ipt_snds, fxtype)
    return ret



def apply_fx(ipt_snds, fxtype, device='cpu'):
    ret = None
    if fxtype == AugType.lpf:
        ret = apply_lpf(ipt_snds, freq=100, q = 0.707)
    elif fxtype == AugType.hpf:
        ret = apply_lpf(ipt_snds, freq=5000, q = 0.707)
    elif fxtype == AugType.bpf:
        ret = apply_bpf(ipt_snds, freq=600, q = 0.707)
    elif fxtype == AugType.rvb:
        ret = apply_reverb(ipt_snds)
    elif fxtype ==  AugType.n10:
        ret = apply_noise(ipt_snds,ntype="pink",snr=10)
    elif fxtype ==  AugType.n5:
        ret = apply_noise(ipt_snds,ntype="pink",snr=5)
    
    return ret

def apply_fx_handler(ipt_snds, fxtype, rng):
    ret = None
    if fxtype == AugType.random:
        ret = apply_random_fx(ipt_snds, rng)
    else:
        ret =  apply_fx(ipt_snds, cur_at)
    return ret
