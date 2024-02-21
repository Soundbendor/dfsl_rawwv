import os,sys,csv
sys.path.insert(0, os.path.dirname(os.path.split(__file__)[0]))
import util.globals as UG
import pandas as pd

csvf = os.path.join(UG.DEF_TAUUAS19DIR , "meta.csv")

df = pd.read_csv(csvf, delimiter="\t")

