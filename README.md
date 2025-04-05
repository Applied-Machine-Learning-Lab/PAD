# PAD
Code for paper '*PAD: Large Language Models Enhancing Sequential Recommendation*' accepted to SIGIR 2025 Full Paper. We adopt SASRec as the backbone model and Prime Pantry dataset as an example. After preparation and phase 0 is done, the overall experiment will take about 20 minutes on 4 Tesla V100 GPUs.

**Preparation**

We use the environment with python 3.9.19 + torch 2.0.1, and install [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) and [LLM2Vec](https://github.com/McGill-NLP/llm2vec/tree/main) by running the following code.
```
cd Transfer-Learning-Library-master
pip install -r requirements.txt
python setup.py install
cd llm2vec
pip install -e .
pip install info-nce-pytorch
```

**Phase 0**

Obtain item embedding through LLM2Vec. Run the following code and save the output embedding in dataset/Amazon_Prime_Pantry_llm2vec.pt.
```
python create_llm2vec_amazon.py
```

**Phase 1**

Pre-train the ID tower with ID embedding.
```
python train_amazon_id_pantry.py
```

**Phase 2**

Train the Alignment tower with rec-guided alignment objective.
```
python train_amazon_id_pantry_transfer.py
```

**Phase 3**

Three-tower Fine-tune.
```
python train_amazon_id_pantry_3tower.py
```
