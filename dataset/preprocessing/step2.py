import pandas as pd
import gzip
import json
import numpy as np

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

c=pd.read_csv('/dataset/amazon/Electronics_cleaned.csv')
#c=c.dropna(how='any')
a=c['user']
u=set(a)
len(u)
a=c['item']
u=set(a)
len(u)

df3 = getDF('/dataset/amazon/meta_Electronics.json.gz')
#All_Amazon_Meta
df=df3[['asin','title']]
df.drop_duplicates(subset=['asin'],keep='last',inplace=True)
#df=df.dropna(how='any')
a=a.to_list()
b = df['asin'].isin(a)
b=b.to_list()
df['new']=b
df=df[(df['new']==True)]
df=df.drop('new', axis=1)
df = df.reset_index(drop=True)
df.shape
df.to_csv('/dataset/amazon/metadata_Electronics.csv',index=False)


