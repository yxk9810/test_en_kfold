#coding:utf-8
import sys
import json
error_cnt =0
from collections import Counter
labels = []
dataset = [json.loads(line.strip()) for line in open('all_data.jsonl','r',encoding='utf-8').readlines()]

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
    train_writer = open('/kaggle/working/train_fold_'+str(idx),'a+',encoding='utf-8')
    for d in train_data:
        train_writer.write(json.dumps(d,ensure_ascii=False)+'\n')
    train_writer.close()
    dev_writer  =open('/kaggle/working/dev_fold_'+str(idx),'a+',encoding='utf-8')
    for d in dev_data:
        dev_writer.write(json.dumps(d,ensure_ascii=False)+'\n')
    dev_writer.close()
    print(len(train_data))
    print(len(dev_data))