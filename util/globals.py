import os

DEF_BASEDIR = os.path.split(os.path.split(__file__)[0])[0] # assumes in subfolder
DEF_EXTDIR = os.path.join(os.sep, 'media','dxk','TOSHIBA EXT')
DEF_DATADIR = os.path.join(DEF_EXTDIR, 'ds') 
DEF_ESC50DIR = os.path.join(DEF_DATADIR, 'ESC-50-master') 
DEF_TINYSOLDIR = os.path.join(DEF_DATADIR, 'TinySOL', 'TinySOL') 
DEF_SAVEDIR = os.path.join(os.sep, 'media', 'dxk', 'TOSHIBA EXT', 'fscil', 'dfsl_rawwv', 'save') 
#DEF_SAVEDIR = os.path.join(DEF_BASEDIR, "save")
DEF_GRAPHDIR = os.path.join(DEF_BASEDIR, "graph")
DEF_RESDIR = os.path.join(DEF_BASEDIR, "res")
DEF_NEP_API = os.path.join(DEF_BASEDIR, "nepapi.txt")
DEF_SEED = 3


