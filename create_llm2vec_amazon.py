
import pandas as pd
import torch
from tqdm import tqdm
from llm2vec import LLM2Vec
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3,4,5,6,7"

l2v = LLM2Vec.from_pretrained(
    "/llama3-8B",
    peft_model_name_or_path="/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
d=pd.read_csv('/dataset/metadata_Prime_Pantry.csv')

item_word_embs=[torch.zeros(4096)]
for i in tqdm(range(8)):
    strlist=d.iloc[1000*i:1000*(i+1),1].tolist()
    item_feature = l2v.encode(strlist)
    item_word_embs.extend(item_feature)

strlist=d.iloc[8000:,1].tolist()
item_feature = l2v.encode(strlist)
item_word_embs.extend(item_feature)

a = torch.stack(tensors=item_word_embs, dim=0)
torch.save(a, '/dataset/Amazon_Clothing_Shoes_and_Jewelry_llm2vec.pt')