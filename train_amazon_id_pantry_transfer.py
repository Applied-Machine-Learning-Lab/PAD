import os

root_data_dir = '/'

dataset = 'amazon'
behaviors = 'amazon'
news = 'amazon'
logging_num = 4
testing_num = 1

bert_model_load = 'llm'
freeze_paras_before = 0
news_attributes = 'title'

mode = 'train'
item_tower = 'modal_cat'

epoch = 400
load_ckpt_name = 'None'

num_workers = 8
transformer_block = 5
l2_weight_list = [0.1]
drop_rate_list = [0.1]
batch_size_list = [16]
embedding_dim_list = [128]
gamma_list=[1e-4]

mo_dnn_layers_list =[4]
dnn_layers_list = [0]

for l2_weight in l2_weight_list:
    for batch_size in batch_size_list:
        for drop_rate in drop_rate_list:
            for embedding_dim in embedding_dim_list:
                for mo_dnn_layers in mo_dnn_layers_list:
                    for dnn_layers in dnn_layers_list:
                        #for i in range(1):
                        for lr in gamma_list:
                            if embedding_dim==512:
                                lr=1e-5
                                load_ckpt_name='epoch-24.pt'
                            elif embedding_dim==128:
                                lr=1e-5
                                load_ckpt_name='epoch-82.pt'
                            elif embedding_dim==64:
                                lr=1e-5
                                load_ckpt_name='epoch-67.pt'
                            label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_Flr{}'.format(
                                item_tower, batch_size, embedding_dim, lr,
                                drop_rate, l2_weight)
                            run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
                                     torchrun --nproc_per_node 8 --master_port 12345\
                                     run_amazon_Prime_Pantry_transfer.py --root_data_dir {}  --dataset {} --behaviors {} --news {}\
                                     --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
                                     --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {} \
                                     --news_attributes {} --bert_model_load {}  " \
                                     "--epoch {} --freeze_paras_before {} " \
                                     "--mo_dnn_layers {} --dnn_layers {} --gamma {} --num_workers {}  --transformer_block {}".format(
                                root_data_dir, dataset, behaviors, news,
                                mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
                                l2_weight, drop_rate, batch_size, lr, embedding_dim,
                                news_attributes, bert_model_load, epoch, freeze_paras_before,
                                mo_dnn_layers, dnn_layers, 0.2, num_workers, transformer_block)
                            os.system(run_py)
