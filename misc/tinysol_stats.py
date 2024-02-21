import os,sys,csv
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
import util.globals as UG
import pandas as pd

inst_family = ["Brass", "Strings", "Winds"]
ret = []

brass_dir = os.path.join(UG.DEF_TINYSOLDIR, "Brass")
str_dir = os.path.join(UG.DEF_TINYSOLDIR, "Strings")
winds_dir = os.path.join(UG.DEF_TINYSOLDIR, "Winds")
out_f = os.path.join(UG.DEF_TINYSOLDIR, "tinysol.csv")

df = pd.read_csv(out_f)
inst = list(set(list(df['instrument'].values)))

