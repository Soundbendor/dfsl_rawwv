def calc_outsize(insize, ksize=3, stride = 1, pad = 0, dilation = 1):
    ret =  int((insize + (2. * pad) - (dilation * (ksize - 1.)) - 1.)/stride) + 1.
    return ret

def calc_insize(outsize, ksize=3, stride = 1, pad = 0, dilation = 1):
    ret = int(((outsize - 1.) * stride) + 1. + (dilation * (ksize - 1.)) - (2. * pad))
    return ret

def calc_padsize(sz, ksize=3, stride=1, dilation=1 ):
    ret = int(0.5 * ((sz  * (stride - 1)) - stride + 1 + (dilation * (ksize - 1))))
    return ret

def insize_by_blocks(num_blocks):
    sz = 1
    for i in range(num_blocks):
        sz = calc_insize(sz, ksize = 3, stride = 3)

    return sz

def insize_by_blocks2(blk_list):
    # (num, ksize, stride)
    sz = 1
    for (cnum, cks, cstr) in blk_list:
        for _ in range(cnum):
            sz = calc_insize(sz, ksize = cks, stride = cstr)
    return sz


def samp_to_sec(samp, sr=44100):
    return samp/float(sr)

def sec_to_samp(sec, sr=44100):
    return sec * float(sr)

def get_sr(samp,sec):
    return float(samp)/float(sec)

def calc_numblocks(sec, sr=44100):
    sz = 1
    gotsec = 0
    nb = 0
    while gotsec < sec:
        sz = calc_insize(sz, ksize = 3, stride = 3)
        gotsec = samp_to_sec(sz, sr=sr)
        nb += 1

    return (nb, sz)
