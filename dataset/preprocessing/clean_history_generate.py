import pandas as pd
d=pd.read_csv('Electronics_cleaned.csv')
frequencies = d['user'].value_counts()
u4=d['user']
u4=u4.to_list()
e=list(set(u4))
data = {'user':e}
data=pd.DataFrame(data)
data.sort_values(by='user', inplace=True, ascending=True)
e=data['user'].to_list()


df={'user':['' for i in range(len(e))],'history':[[] for i in range(len(e))]}
df = pd.DataFrame(df)
#df = pd.DataFrame(columns=['user', 'history'])
    
start=0
for k in tqdm(range(len(e))):
    i=e[k]
    l=frequencies[i]
    #data = {'user': i, 'history': d.iloc[start:start+l,0].tolist()}
    #data_series = pd.Series(data)
    #df = pd.concat([df, pd.DataFrame([data_series])], ignore_index=True)
    df.iloc[k,0]=i
    df.iloc[k,1]=d.iloc[start:start+l,0].tolist()
    start=start+l

df.to_csv('Electronics_cleaned_history.csv')

