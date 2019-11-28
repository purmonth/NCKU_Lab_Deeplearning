import pandas as pd
import numpy as np
import json
import glob
import csv
import os

li = []
df = pd.read_json("json/2019-01-02.json",orient = 'index')
df = pd.DataFrame(df,index=["0050"],columns=['high'])
number = 0

filenames = glob.glob("json/2019-*.json")
for filename in filenames:
    df1 = pd.read_json(filename,orient = 'index')
    df1 = pd.DataFrame(df1,index=["0050"],columns=['high'])
    df = pd.DataFrame(pd.np.append(df, df1), columns=['0050_high'])
    print(filename)


print(df)
df.to_csv('0050_2019_high.csv',header=0,index=0)

##pd.DataFrame(pd.np.append(data1.values, data2.values), columns=['A'])
