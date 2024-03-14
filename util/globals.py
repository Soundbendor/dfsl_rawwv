import os

DEF_ROOTDIR = os.path.split(os.path.split(__file__)[0])[0] # assumes in subfolder
DEF_EXTDIR = os.path.join(os.sep, 'media','dxk','TOSHIBA EXT')
DEF_DATADIR = os.path.join(DEF_EXTDIR, 'ds') 
DEF_ESC50DIR = os.path.join(DEF_DATADIR, 'ESC-50-master') 
DEF_TINYSOLDIR = os.path.join(DEF_DATADIR, 'TinySOL', 'TinySOL') 
DEF_AUDIOSETDIR = os.path.join(DEF_DATADIR, 'audioset', 'ksrc') 
DEF_US8KDIR = os.path.join(DEF_DATADIR, 'urbansound8k') 
DEF_TAUUAS19DIR = os.path.join(DEF_DATADIR, 'tau_urban_acoustic_scenes_2019', 'TAU-urban-acoustic-scenes-2019-development') 
DEF_SAVEDIR = os.path.join(os.sep, 'media', 'dxk', 'TOSHIBA EXT', 'fscil', 'dfsl_rawwv', 'save') 
#DEF_SAVEDIR = os.path.join(DEF_ROOTDIR, "save")
DEF_GRAPHDIR = os.path.join(DEF_ROOTDIR, "graph")
DEF_RESDIR = os.path.join(DEF_ROOTDIR, "res")
DEF_NEP_API = os.path.join(DEF_ROOTDIR, "nepapi.txt")
DEF_SEED = 3
DEF_BASEDIR =DEF_ESC50DIR
DEF_NOVELDIR =DEF_ESC50DIR

