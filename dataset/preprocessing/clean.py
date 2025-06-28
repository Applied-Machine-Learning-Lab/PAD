
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



import numpy as np
#import tqdm
from tqdm import tqdm
import pandas as pd

c=pd.read_csv('/dataset/amazon/Electronics.csv',names=['item','user','rating','timestamp'])
a=c['item']
df3 = getDF('/dataset/amazon/meta_Electronics.json.gz')
#All_Amazon_Meta
df=df3[['asin','title']]
df.drop_duplicates(subset=['asin'],keep='last',inplace=True)
a=df['asin']
a=a.to_list()
b = c['item'].isin(a)
b=b.to_list()
c['new']=b
c=c[(c['new']==True)]
c=c.drop('new', axis=1)
c = c.reset_index(drop=True)
c.to_csv('/dataset/amazon/Electronics.csv',index=False)


c=pd.read_csv('Electronics.csv')
c=c.dropna(how='any')
c=c[(c['rating']>=4)]
c.sort_values(by=['user', 'timestamp'], inplace=True, ascending=[True, True])

frequencies = c['user'].value_counts()
 
d = c[c['user'].isin(frequencies[frequencies >=5].index)]
frequencies = d['user'].value_counts()
u4=d['user']
u4=u4.to_list()
#u=set(u4)
d['new']=0
d = d.reset_index(drop=True)

e=list(set(u4))
data = {'user':e}
data=pd.DataFrame(data)
data.sort_values(by='user', inplace=True, ascending=True)
e=data['user'].to_list()

start=0

for k in tqdm(range(len(e))):
    i=e[k]
    l=frequencies[i]
    if l<=23:
        d.iloc[start:start+l,4]=1
    else:
        d.iloc[start+l-22:start+l,4]=1
    start=start+l

dd=d[(d['new']==1)]
#c=c.dropna(how='any')
dd=dd.drop('new', axis=1)
dd.to_csv('/dataset/amazon/Electronics_cleaned.csv',index=False)


