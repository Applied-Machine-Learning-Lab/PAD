python


import torch
from tqdm import tqdm
from llm2vec import LLM2Vec
import os
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"

l2v = LLM2Vec.from_pretrained(
    "llama3-8B",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    device_map="cuda:1",
    torch_dtype=torch.bfloat16,
)
d=pd.read_csv('/dataset/amazon/metadata_Clothing_Shoes_and_Jewelry.csv')
d['title']=d['title'].astype(str)

#item_word_embs=[torch.zeros(4096)]
item_word_embs=[]
for i in tqdm(range(600,1200)):
    strlist=d.iloc[1000*i:1000*(i+1),1].tolist()
    item_feature = l2v.encode(strlist)
    item_word_embs.extend(item_feature)

a = torch.stack(tensors=item_word_embs, dim=0)
torch.save(a, '/Amazon_Clothing_Shoes_and_Jewelry_llm2vec_'+str(2)+'.pt')

exit()

import torch
from tqdm import tqdm
from llm2vec import LLM2Vec
import os
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3"

l2v = LLM2Vec.from_pretrained(
    "llama3-8B",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    device_map="cuda:1",
    torch_dtype=torch.bfloat16,
)
d=pd.read_csv('/dataset/amazon/metadata_Clothing_Shoes_and_Jewelry.csv')
d['title']=d['title'].astype(str)
#item_word_embs=[torch.zeros(4096)]
item_word_embs=[]

for i in tqdm(range(1200,1558)):
    strlist=d.iloc[1000*i:1000*(i+1),1].tolist()
    item_feature = l2v.encode(strlist)
    item_word_embs.extend(item_feature)

strlist=d.iloc[1000*(800+758):,1].tolist()
item_feature = l2v.encode(strlist)
item_word_embs.extend(item_feature)


a = torch.stack(tensors=item_word_embs, dim=0)
torch.save(a, '/Amazon_Clothing_Shoes_and_Jewelry_llm2vec_'+str(3)+'.pt')
exit()


