import pandas as pd
from xsec_collector import main as xsec
from pixel_collector import main as pixel

xy = xsec()
z = pixel()
xyz = pd.concat([xy, z], axis=1)
print(xyz)
# print(xyz['X'])