import pandas as pd
from xsec_collector import main as xsec
from pixel_collector import main as pixel
import csv

xy = xsec()
z = pixel()
xyz = pd.concat([xy, z], axis=1)
print(xyz)
# print(xyz['X'])

# xyz.to_csv('./xsec.csv', sep=',', na_rep='NaN')
