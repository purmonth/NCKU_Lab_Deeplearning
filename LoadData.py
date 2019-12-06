import pandas as pd
import numpy as np
import json
import glob
import csv
import os

li = []
df = pd.DataFrame(columns=['close', 'date'])
##df = pd.read_json("json/2019-01-02.json",orient = 'index')
##df = pd.DataFrame(df,index=["0050"],columns=['close'])
number = 0

filenames = glob.glob("json/2019-01-0*.json")
for filename in filenames:
    df1 = pd.read_json(filename,orient = 'index')
    ##print(df1)
    df1 = pd.DataFrame(df1,index=["0050"],columns=['close'])
    filename = filename.replace(".json","")
    df1['date'] = filename.replace("json/","")
    print(df1)
    df = df.append(df1)


print(df)
df.to_csv('test.csv',header=0,index=0)

##pd.DataFrame(pd.np.append(data1.values, data2.values), columns=['A'])
