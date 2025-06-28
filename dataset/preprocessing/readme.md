# Procedure of data preprocessing for Amazon

**Step 1**

Run clean.py to generate xxx_clean.csv

**Step 2**

Run step2.py to generate metadata_xxx.csv

**Step 3**

Run amazon1.sh to generate 3 xxx.pt files

**Step 4**

Refer to read_news_bert_amazon and read_behaviors_amazon function in preprocess_amazon.py to generate xxx_fre.csv

**Step 5**

Run clean_history_generate.py to generate xxx_cleaned_history.csv
