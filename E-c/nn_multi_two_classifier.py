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

classes=["love", "sadness", "fear", "joy", "trust", "optimism", "disgust", "pessimism", "anger", "anticipation", "surprise"]
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
    label3 = []
    label4 = []
    label5 = []
    label6 = []
    label7 = []
    label8 = []
    label9 = []
    label10 = []
    label11 = []
    Weight=[0,0,0,0,0,0,0,0,0,0,0]
    f = open(filename, encoding='utf-8')  #设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    train_data = json.load(f)
    for i in train_data:
        if i==train_data[0]:
            continue
        data.append(i['bert_vector'])
        label1.append(float(i['love']))
        if float(i['love'])==1:
            Weight[0]+=1
        label2.append(float(i['sadness']))
        if float(i['sadness'])==1:
            Weight[1]+=1
        label3.append(float(i['fear']))
        if float(i['fear'])==1:
            Weight[2]+=1
        label4.append(float(i['joy']))
        if float(i['joy'])==1:
            Weight[3]+=1
        label5.append(float(i['trust']))
        if float(i['trust'])==1:
            Weight[4]+=1
        label6.append(float(i['optimism']))
        if float(i['optimism'])==1:
            Weight[5]+=1
        label7.append(float(i['disgust']))
        if float(i['disgust'])==1:
            Weight[6]+=1
        label8.append(float(i['pessimism']))
        if float(i['pessimism'])==1:
            Weight[7]+=1
        label9.append(float(i['anger']))
        if float(i['anger'])==1:
            Weight[8]+=1
        label10.append(float(i['anticipation']))
        if float(i['anticipation'])==1:
            Weight[9]+=1
        label11.append(float(i['surprise']))
        if float(i['surprise'])==1:
            Weight[10]+=1
    Label.append([label1,label2,label3,label4,label5,label6,label7,label8,label9,label10,label11])
    print(Weight)
    return data,Label,Weight

print('加载训练集……')
x ,y,Weight=loadJson(r"train_new-vector.json")
x=np.array(x)
x=x[:,0,:]
y=np.array(y[0])
y=y.swapaxes(1,0)
#x=np.concatenate([x,a], axis = 1)
X = torch.from_numpy(np.array(x)).type(torch.FloatTensor)
y = torch.from_numpy(np.array(y)).type(torch.LongTensor)


print('加载dev集……')
x_valid ,y_valid=loadJson(r"dev_new-vector.json")[0:2]
x_valid=np.array(x_valid)
x_valid=x_valid[:,0,:]
y_valid=np.array(y_valid[0])
y_valid=y_valid.swapaxes(1,0)
#x_valid=np.concatenate([x_valid,c], axis = 1)
x_valid = torch.from_numpy(np.array(x_valid)).type(torch.FloatTensor)
y_valid = torch.from_numpy(np.array(y_valid)).type(torch.LongTensor)
#print(x_valid)
#print(y_valid)

print('加载测试集……')
x_test ,y_test=loadJson(r"test_new-vector.json")[0:2]
tmp=y_test
x_test=np.array(x_test)
x_test=x_test[:,0,:]
y_test=np.array(y_test[0])
y_test=y_test.swapaxes(1,0)
#x_test=np.concatenate([x_test,b], axis = 1)
x_test = torch.from_numpy(np.array(x_test)).type(torch.FloatTensor)
y_test = torch.from_numpy(np.array(y_test)).type(torch.LongTensor)


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
            if t[0] < t[1]:
                ans.append(1)
            else:
                ans.append(0)
        return torch.tensor(ans)

def training(y,X,y_test,x_test,x_valid,y_valid,Weight):
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
                return testing(model, y_test, x_test)
    return testing(model,y_test,x_test)

def testing(model,y_test,x_test):
    #print("testing")
    model=model.eval()
    prediction = model.predict(x_test)
    num = 0
    wrong = 0
    right = 0
    should_be_1=0
    one_to_1=0
    one_to_0=0
    zero_to_1=0
    result.append(np.array(prediction).tolist())
    for x in prediction:
        if x == y_test[num]:
            if y_test[num] == 1:
                one_to_1 += 1
                should_be_1 += 1
            right += 1
        else:
            if y_test[num] == 1:
                should_be_1 += 1
                one_to_0 += 1
            if y_test[num] == 0:
                zero_to_1+=1
            wrong += 1
        num += 1

    #print(y_test.sum())
    #print(should_be_1, one_to_1)
    print("准确度：",right * 1.0 / (wrong + right))
    print("presicion:",one_to_1/(zero_to_1+one_to_1))
    print("recall:",one_to_1/should_be_1)
    return right,wrong,should_be_1,one_to_1,zero_to_1

def calculate_micro_F1_score(should_be_1,one_to_1,zero_to_1):
    micro_P=sum(one_to_1)/(sum(zero_to_1)+sum(one_to_1))
    micro_R=sum(one_to_1)/sum(should_be_1)
    F=2*micro_P*micro_R/(micro_P+micro_R)
    return F

def calculate_Macro_F1(should_be_1,one_to_1,zero_to_1):
    should_be_1=np.array(should_be_1)
    one_to_1=np.array(one_to_1)
    zero_to_1=np.array(zero_to_1)
    P=one_to_1/(zero_to_1+one_to_1)
    R=one_to_1/should_be_1
    F=2*P*R/(P+R)
    macro=F.sum()/11
    return macro

all_right=[]
all_wrong=[]
should_be_1=[]
one_to_1=[]
zero_to_1=[]
for k in range(11):
    print(classes[k]+" is being trained and tested!!")
    batch_size = 64
    train_dataset = TensorDataset(X, y[:,k])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    right,wrong,should_be_1_,one_to_1_,zero_to_1_=training(y[:,k],X,y_test[:,k],x_test,x_valid,y_valid[:,k],Weight[k])
    all_right.append(right)
    all_wrong.append(wrong)
    should_be_1.append(should_be_1_)
    one_to_1.append(one_to_1_)
    zero_to_1.append(zero_to_1_)


print('F:',calculate_micro_F1_score(should_be_1,one_to_1,zero_to_1))
print('macro-F',calculate_Macro_F1(should_be_1,one_to_1,zero_to_1))

result2=[]
result3=[]
for i in range(len(result[0])):
    result1=[]
    x=[]
    for j in range(11):
        result1.append(result[j][i])
        x.append(int(tmp[0][j][i]))
    result2.append(result1)
    result3.append(x)

acc=0
all=0
for i in range(len(result2)):
    for j in range(11):
        if result2[i][j] == 0 and result3[i][j] == 0:
            continue
        else:
            all += 1
            if result2[i][j] == 1 and result3[i][j] == 1:
                acc += 1
print(acc/all)

f2=open('result.tsv','w',encoding='utf-8')
text2='idx\tpred_label\n'
f2.write(text2)
for i in range(len(result2)):
    f2.write(str(i)+'\t'+str(result2[i])+'\n')






