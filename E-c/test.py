#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:nn_multi_two_classifier.py
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import setiWord
from gensim.models.word2vec import Word2Vec
from torchvision import datasets,transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset,TensorDataset

classes=[ "joy",  "anger"]
result=[]

sentiWordvector=np.load('train.npy')
a=sentiWordvector[0:6838]
b=sentiWordvector[6838:10097]
c=sentiWordvector[10097:10983]

def loadJson(filename):
    #global data_rvt
    data=[]
    Label=[]
    label1=[]
    label2=[]
    Weight=[0,0]
    f = open(filename, encoding='utf-8')  #设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    train_data = json.load(f)
    for i in train_data:
        if i==train_data[0]:
            continue
        data.append(i['bert_vector'])

        label1.append(float(i['joy']))
        if float(i['joy'])==1:
            Weight[0]+=1
        label2.append(float(i['anger']))
        if float(i['anger'])==1:
            Weight[1]+=1
    Label.append([label1,label2])
    print(Weight)
    return data,Label,Weight

def loadJson2(filename):
    #global data_rvt
    data=[]
    f = open(filename, encoding='utf-8')  #设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    train_data = json.load(f)
    for i in train_data:
        data.append(i['bert_vector'])
    return data

print('加载训练集……')
x ,y,Weight=loadJson(r"train_new-vector.json")
x=np.array(x)
x=x[:,0,:]
y=np.array(y[0])
y=y.swapaxes(1,0)
#x=np.concatenate([x,a], axis = 1)
X = torch.from_numpy(np.array(x)).type(torch.FloatTensor)
y = torch.from_numpy(np.array(y)).type(torch.LongTensor)

x_test ,y_test,Weight2=loadJson(r"test_new-vector.json")
x_test=np.array(x_test)
x_test=x_test[:,0,:]
y_test=np.array(y_test[0])
y_test=y_test.swapaxes(1,0)
#x_test=np.concatenate([x_test,b], axis = 1)
x_test = torch.from_numpy(np.array(x_test)).type(torch.FloatTensor)
y_test = torch.from_numpy(np.array(y_test)).type(torch.LongTensor)

X=torch.cat((X,x_test),0)
y=torch.cat((y,y_test),0)
print(X.shape)
print(y.shape)

print('加载测试集……')
x_test =loadJson2(r"dev_new-vector.json")[0:2]
x_test=np.array(x_test)
x_test=x_test[:,0,:]
#x_test=np.concatenate([x_test,b], axis = 1)
x_test = torch.from_numpy(np.array(x_test)).type(torch.FloatTensor)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, 200)
        self.fc2 = nn.Linear(200, 20)
        #self.fc3 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(20, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc3(x)
        #x = self.dropout(x)
        #x = F.relu(x)

        #x = self.fc4(x)
        return x

    def predict(self, x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0] <= t[1]:
                ans.append(1)
            else:
                ans.append(0)
        return torch.tensor(ans)

def training(y,X,x_test,Weight):
    Weight1=torch.from_numpy(np.array([Weight/(len(y)),(len(y)-Weight)/(len(y))])).type(torch.FloatTensor)
    #print(Weight1,len(y))
    model = Net()
    criterion = nn.CrossEntropyLoss(weight=Weight1)  # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam梯度优化器
    #print("training")
    epoch = 300
    for j in range(epoch):
        for i, data in enumerate(train_loader):
            inputs, outputs = data
            inputs = Variable(inputs, requires_grad=True)  # .to('cuda:0')
            outputs = Variable(outputs)  # .to('cuda:0')
            #print("epoch:", j, " i:", i, "inputs", inputs.data.size(), "labels", outputs.data.size())
            pred_y = model(inputs)
            loss = criterion(pred_y, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('loss:', loss)
            #if j % 5 == 0:
                #print(loss.mean())
            if loss.mean()<=0.005:
                return testing(model, x_test)
    return testing(model,x_test)

def testing(model,x_test):
    #print("testing")
    model=model.eval()
    prediction = model.predict(x_test)
    result.append(np.array(prediction).tolist())
    print(result)

for k in range(2):
    print(classes[k]+" is being trained and tested!!")
    batch_size = 64
    train_dataset = TensorDataset(X, y[:,k])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    training(y[:,k],X,x_test,Weight[k])


result2=[]
for i in range(len(result[0])):
    result1=[]
    for j in range(2):
        result1.append(result[j][i])
    result2.append(result1)

f2=open('result.tsv','w',encoding='utf-8')
text2='idx\tpred_label\n'
f2.write(text2)
for i in range(len(result2)):
    f2.write(str(i)+'\t'+str(result2[i])+'\n')






