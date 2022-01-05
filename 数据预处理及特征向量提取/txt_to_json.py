#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:txt_to_json.py
import json
import os

def txt2json(filename):
    f = open(filename, encoding='utf-8')
    data = f.readlines()
    list=[]
    line_num=0
    for line in data:
        line_num=line_num+1
        if line_num==1:
            continue
        start=0
        end=0
        k=0
        info=["ID","Tweet",	"valence","intensity"]
        dataset={"ID":"","Tweet":"","valence":"","intensity":""}
        #print(line)
        for j in range(len(line)):
            if line[j]=="\n" :
                #dataset[info[k]] = line[start:j]
                for i in range(start,len(line)):                     #读取标签只保留数值
                    if line[i]=="\n" or line[i]==":":
                        dataset[info[k]] = line[start:i]
                        break
                #print(dataset[info[k]])
                break
            if line[j]=="\t" :
                dataset[info[k]]=line[start:end]
                #print(dataset[info[k]])
                start=end+1
                k+=1
            end+=1

        #print(dataset)
        list.append(dataset)
    with open(filename[:-4] + ".json", "w") as f:
        json.dump(list, f)

path1="C:\\Users\\user\\PycharmProjects\\bert\\data\\V-oc"
path2="C:\\Users\\user\\PycharmProjects\\bert\\data\\V-reg"
files= os.listdir(path1) #得到文件夹下的所有文件名称
for file in files: #遍历文件夹
    txt2json(path1+"\\"+file)

files= os.listdir(path2) #得到文件夹下的所有文件名称
for file in files: #遍历文件夹
    txt2json(path2+"\\"+file)

'''files= os.listdir(path2) #得到文件夹下的所有文件名称
for subpath in files: #遍历文件夹
    subfiles = os.listdir(path2+"\\"+subpath)
    for file in subfiles:
        txt2json(path2+"\\"+subpath+"\\"+file)'''

#files= os.listdir(path2) #得到文件夹下的所有文件名称
#for file in files: #遍历文件夹
#     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
#        txt2json(path2+"\\"+file)