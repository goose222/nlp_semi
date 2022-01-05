import tensorflow as tf
import os
import json
from bert_serving.client import BertClient

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#print(tf.__version__)
#cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
#print(gpus, cpus)

# bert-serving-start -model_dir C:\Users\user\PycharmProjects\bert\bert-base-uncased -num_worker=1 -max_seq_len=64


path = 'C:\\Users\\user\\PycharmProjects\\predata\\E-C-no-emoji'
filenames = ['test_new.json']

for file in filenames:
    f = open(path + '\\' + file, 'r', encoding='utf-8')
    print(file)
    data = json.load(f)
    filelist=[]
    i=0
    #print(i)
    for line in data:
        i=i+1
        if i%100==0 :
            print(i)
        dataset = {"anger": "", "anticipation": "", "disgust": "", "fear": "", "joy": "", "love": "", "optimism": "", "pessimism": "", "sadness": "", "surprise": "", "trust": "", "bert_vector": ""}
        # vals = line['intensity']
        text = line['Tweet']
        bc = BertClient()
        vector = bc.encode([text])
        # dataset['intensity'] = vals
        dataset['anger'] = line['anger']
        dataset['anticipation'] = line['anticipation']
        dataset['disgust'] = line['disgust']
        dataset['fear'] = line['fear']
        dataset['joy'] = line['joy']
        dataset['love'] = line['love']
        dataset['optimism'] = line['optimism']
        dataset['pessimism'] = line['pessimism']
        dataset['sadness'] = line['sadness']
        dataset['surprise'] = line['surprise']
        dataset['trust'] = line['trust']
        dataset['bert_vector'] = vector.tolist()
        filelist.append(dataset)
    with open(file[:-5] + '-vector' + ".json", "w") as f:
        json.dump(filelist, f)
    f.close()

#with open(train, encoding='utf-8') as f:
    '''data = json.load(f)
    trainlist=[]
    i=0
    #print(i)
    for line in data:
        i=i+1
        if i%100==0 :
            print(i)
        dataset = {"intensity": "", "bert_vector": ""}
        vals = line['intensity']
        text = line['Tweet']
        bc = BertClient()
        vector = bc.encode([text])
        dataset['intensity'] = vals
        dataset['bert_vector'] = vector.tolist()
        trainlist.append(dataset)
    with open(train[:-5] + '-vector' + ".json", "w") as f:
        json.dump(trainlist, f)
    f.close()'''


#with open(test, encoding='utf-8') as f:
    '''data = json.load(f)
    testlist=[]
    i=0
    for line in data:
        i=i+1
        if i%100==0 :
            print(i)
        dataset = {"intensity": "", "bert_vector": ""}
        vals = line['intensity']
        text = line['Tweet']
        bc = BertClient()
        vector = bc.encode([text])
        dataset['intensity'] = vals
        dataset['bert_vector'] = vector.tolist()
        testlist.append(dataset)
    with open(test[:-5] + '-vector' + ".json", "w") as f:
        json.dump(testlist, f)
    f.close()'''


