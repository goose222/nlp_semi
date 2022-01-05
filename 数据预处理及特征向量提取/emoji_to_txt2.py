#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:emoji_to_txt.py
import json
import os

emoji_list=[]
txt_list=[]
def get_emoji_list():
    f = open("emoji.txt", encoding='utf-8')
    txt=f.readlines()
   # print(txt[0],1)
    for i in txt:
        if len(i)>68:
            k=1
            while(i[k+67]!=' '):
                k+=1
            #print(i[67:(67+k)])
            emoji_list.append(i[67:(67+k)])
            x=68+k
            while (i[x] != ' '):
                x += 1
            txt_list.append(i[x+1:-1])
    #print(len(emoji_list))
    #print(len(txt_list))


def find_emoji(emoji):
    for j in range(len(emoji_list)):
        if emoji==emoji_list[j]:
            return j
    return -1

get_emoji_list()

'''path = 'C:\\Users\\user\\PycharmProjects\\predata\\E-C'
filenames = os.listdir(path)

for file in filenames:
    txt_set=[]
    f = open(path+'\\'+file,'r', encoding='utf-8')  #设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
    train_data = json.load(f)
    for i in train_data:
        txt_set.append({'Tweet':i['Tweet'],"anger": i['anger'], "anticipation": i['anticipation'], "disgust": i['disgust'], "fear": i['fear'], "joy": i['joy'], "love": i['love'], "optimism": i['optimism'], "pessimism": i['pessimism'], "sadness": i['sadness'], "surprise": i['surprise'], "trust": i['trust']})

    txt_set_new=[]
    t = 0
    for k in  txt_set:
        t = t+1
        txt=k["Tweet"]
        # print(txt)
        i = 0
        while (i < len(txt)):
            if (find_emoji(txt[i]) != -1):
                x = find_emoji(txt[i])
                # print(i)
                txt = txt[:i] + " " + txt_list[x] + " " + txt[i + 1:]
            i = i + 1
        #print(txt)
        # print(t)
        txt_set_new.append({'Tweet':txt,"anger": k['anger'], "anticipation": k['anticipation'], "disgust": k['disgust'], "fear": k['fear'], "joy": k['joy'], "love": k['love'], "optimism": k['optimism'], "pessimism": k['pessimism'], "sadness": k['sadness'], "surprise": k['surprise'], "trust": k['trust']})

    new_file = file[:-5]+"_new.json"
    with open("E-C-no-emoji\\"+new_file, "w") as f:
        json.dump(txt_set_new, f)'''
