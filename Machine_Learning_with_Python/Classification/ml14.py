import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd 

df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace("?", -99999, inplace = True)
df.drop(["id"], 1, inplace = True)
