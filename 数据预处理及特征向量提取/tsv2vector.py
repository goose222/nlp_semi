import tensorflow as tf
import os
import json
from bert_serving.client import BertClient
import emoji_to_txt2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 读入tsv文件，生成所有句子的vector
# 命令行输入这句话先：
# bert-serving-start -model_dir C:\Users\user\PycharmProjects\bert\bert-base-uncased -num_worker=1 -max_seq_len=64
def sen2vec(filename):
    f = open(filename, encoding='utf-8')
    data = f.readlines()
    veclist = []
    i = -1
    for line in data:
        i = i+1
        if i == 0:
            continue
        txt = ''
        for j in range(len(line)):
            if line[j]=="\t" :
                txt = line[j+1:len(line)]
                break
        # 处理表情
        k = 0
        while (k < len(txt)):
            if (emoji_to_txt2.find_emoji(txt[k]) != -1):
                x = emoji_to_txt2.find_emoji(txt[k])
                txt = txt[:k] + " " + emoji_to_txt2.txt_list[x] + " " + txt[k + 1:]
            k = k + 1
        # 生成句向量
        bc = BertClient()
        vector = bc.encode([txt]).tolist()
        veclist.append(vector)

        # with open('vector.json", "w") as f:
        #     json.dump(vector, f)
    return veclist

# print(sen2vec('cls_test.tsv'))