#coding:utf-8
import sys
import json
error_cnt =0
from collections import Counter
labels = []
dataset = []
with open('英文重要性标注-1k.txt','r',encoding='utf-8') as lines:
    for line in lines:
        data = json.loads(line.strip())
        extra_info = data['extra']['英文标题']
        zh_title = data['extra']['中文标题']
        if '英文内容' not in data['extra']:
            #print(data)
            # error_cnt+=1
            continue
        extra_content = data['extra']['英文内容']
        zh_content = data['extra']['中文内容']
        if 'content' in data and data['content']:
            # print(data['content'])
            qa_pairs = json.loads(data['content'])[0]['qa_pairs'][0]['blocks'][0]['text']
            label = int(qa_pairs.replace('重要性','').replace('分',''))
            json_data = {'title':extra_info,'content':extra_content.replace('\u3000','').replace('\u2002',''),'label':label,'zh_title':zh_title,'zh_content':zh_content}
            labels.append(label)
            dataset.append(json_data)
            # print(json.dumps(json_data,ensure_ascii=False))
            # writer.write(json.dumps(json_data,ensure_ascii=False)+'\n')
import random
random.shuffle(dataset)
print(len(dataset))
import numpy as np
def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds
K= 10
folds = kfold_indices(dataset,10)
for idx,fold in enumerate(folds):
    train_indices = fold[0]
    dev_indices = fold[1]
    train_data = []
    dev_data = []
    for index in train_indices:
        train_data.append(dataset[index])
    for index in dev_indices:
        dev_data.append(dataset[index])
    train_writer = open('d:/kfold_en_data/train_fold_'+str(idx),'a+',encoding='utf-8')
    for d in train_data:
        train_writer.write(json.dumps(d,ensure_ascii=False)+'\n')
    train_writer.close()
    dev_writer  =open('d:/kfold_en_data/dev_fold_'+str(idx),'a+',encoding='utf-8')
    for d in dev_data:
        dev_writer.write(json.dumps(d,ensure_ascii=False)+'\n')
    dev_writer.close()
    print(len(train_data))
    print(len(dev_data))