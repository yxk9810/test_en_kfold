import torch
import torch.nn as nn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", default="", type=str, help="")



args = parser.parse_args()

train_filename ='/kaggle/working/'+'train_fold_'+str(args.fold)
dev_filename = '/kaggle/working/'+'dev_fold_'+str(args.fold)
model_dir = '/kaggle/working/models/'

class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


class MultiDSCLoss(torch.nn.Module):
    r"""
    Creates a criterion that optimizes a multi-class Self-adjusting Dice Loss
    ("Dice Loss for Data-imbalanced NLP Tasks" paper)

    Args:
        alpha (float): a factor to push down the weight of easy examples
        gamma (float): a factor added to both the nominator and the denominator for smoothing purposes
        reduction (string): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Shape:
        - logits: `(N, C)` where `N` is the batch size and `C` is the number of classes.
        - targets: `(N)` where each value is in [0, C - 1]
    """

    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.smooth) / (probs_with_factor + 1 + self.smooth)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none" or self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"Reduction `{self.reduction}` is not supported.")
# coding:utf-8
import sys
import re
from transformers import BertModel
from transformers import AutoModel
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import random
import numpy as np

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class BertClassifier(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=config.pretrain_model_path)
        self.dropout = nn.Dropout(0.2)
        self.cls_layer1 = nn.Linear(config.hidden_size, config.class_num)

    def forward(self, input_ids=None, attention_mask=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = torch.mean(bert_output.last_hidden_state, dim=1)
        #         pooled_output = bert_output.last_hidden_state[:,0,:]

        logits = self.dropout(pooled_output)
        output = self.cls_layer1(logits)
        return output


# coding:utf-8
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from transformers import BertTokenizer
from transformers import AutoTokenizer
import torch

is_english = False
checkpoint_name = 'google-bert/bert-base-uncased'
# checkpoint_name = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
class NLPCCTaskDataSet(Dataset):
    def __init__(self, filepath='', is_train=True, mini_test=True, is_test=False):
        self.mini_test = mini_test
        self.is_test = is_test
        self.reply_lens = []
        self.dataset = self.load_json_data(filepath)

    def load_json_data(self, filename):
        return [json.loads(line.strip()) for line in open(filename, 'r', encoding='utf-8').readlines()]

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def collate_fn_nlpcc(batch, max_seq_lenght=512, name='like'):
    batch_data = []
    batch_labels = []
    for d in batch:
        content = d['content'].replace(' ', '')
        content = re.sub(r'^https?:\/\/.*[\r\n]*', '', content, flags=re.MULTILINE)
        batch_data.append(d['title'] + '[SEP]' + content)
        label = 0
        if 'label' in d:
            label = int(d['label']) - 1
            if label < 0:
                label = 0
        batch_labels.append(label)
    tokens = tokenizer(
        batch_data,
        padding=True,
        max_length=max_seq_lenght,
        truncation=True)
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    labels = torch.tensor(batch_labels, dtype=torch.long)
    return seq, mask, labels


import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs.squeeze(), targets.float())
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


import torch


class FGM():

    def __init__(self, model, embedding_name='word_embeddings.', epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.embedding_name = embedding_name
        self.back_params = {}

    def attack(self):
        """
        对embedding添加扰动(根据梯度下降的反方向，epsilon控制扰动幅度)
        :return:
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                self.back_params[name] = param.data.clone()
                norm = torch.norm(param.grad)

                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # 恢复正常参数
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.embedding_name in name:
                assert name in self.back_params
                param.data = self.back_params[name]
        self.back_params = {}


# define config file
class Config:
    pretrain_model_path = checkpoint_name
    hidden_size = 768
    learning_rate = 5e-5
    epoch = 1
    class_num = 3
    train_file = './data/'
    train_file = train_filename
    dev_file =  dev_filename
    test_file = dev_filename
    target_dir = model_dir
    use_fgm = True


import time

now_time = time.strftime("%Y%m%d%H", time.localtime())
from transformers import AdamW
config = Config()

def train(model, train_data_loader, device, optimizer, fgm=None):
    model.train()
    total_loss, total_accuracy = 0, 0
    for step, batch in enumerate(tqdm(train_data_loader)):
        sent_id, mask, like_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        model.zero_grad()
        logits_like = model(sent_id, mask)
        pos_weight = torch.FloatTensor([6.4])
        loss_fn = nn.CrossEntropyLoss()
        #         loss_fn = MultiFocalLoss(num_class=config.class_num, gamma=2.0, reduction='mean')
        loss = loss_fn(logits_like, like_labels)
        loss.backward()
        if config.use_fgm:
            fgm.attack()
            logits_like = model(sent_id, mask)
            loss_adv = loss_fn(logits_like, like_labels)
            loss_adv.backward()
            fgm.restore()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_item = loss.item()
        total_loss += loss_item
    avg_loss = total_loss / len(train_data_loader)
    return avg_loss


import numpy as np
from sklearn.metrics import f1_score


def evaluate(model, dev_data_loader, device):
    model.eval()
    total_loss, total_accuracy = 0, 0
    gold_like = []
    pred_like = []
    pos_weight = torch.cuda.FloatTensor([6.4])

    for step, batch in enumerate(dev_data_loader):
        #         print(batch[2])
        targets = batch[2].type(torch.LongTensor)
        sent_id, mask, like_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        logits_like = model(sent_id, mask)
        loss_fn = nn.CrossEntropyLoss()
        #         loss_fn = MultiFocalLoss(num_class=config.class_num, gamma=2.0, reduction='mean')
        loss = loss_fn(logits_like, like_labels.view(-1))
        loss_item = loss.item()
        preds = torch.argmax(torch.softmax(logits_like, dim=-1), dim=-1).detach().cpu().numpy()
        gold = batch[2].detach().cpu().numpy()
        gold_like.extend(gold.tolist())
        pred_like.extend(preds.tolist())
        total_loss += loss_item
    avg_loss = total_loss / len(pred_like)
    from sklearn.metrics import classification_report
    golds = [int(d) for d in gold_like]
    preds = [int(p) for p in pred_like]
    print(classification_report(golds, preds))
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    accuracy = accuracy_score(golds, preds)
    return avg_loss, accuracy


def test(model, dev_data_loader):
    model.eval()
    gold_like = []
    pred_like = []
    for step, batch in enumerate(dev_data_loader):
        sent_id, mask, like_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        logits_like = model(sent_id, mask)
        preds = torch.argmax(torch.softmax(logits_like, dim=-1), dim=-1).detach().cpu().numpy()
        gold = batch[2].detach().cpu().numpy()
        gold_like.extend(gold.tolist())
        pred_like.extend(preds.tolist())
    return gold_like, pred_like

from functools import partial

model = BertClassifier(config)
optimizer = AdamW(model.parameters(), lr=config.learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
name = 'like'
dataset = NLPCCTaskDataSet(filepath=config.train_file, mini_test=False)
train_data_loader = DataLoader(dataset, batch_size=16, collate_fn=partial(collate_fn_nlpcc, name=name), shuffle=True)
dev_dataset = NLPCCTaskDataSet(filepath=config.dev_file, mini_test=False, is_test=False)
dev_data_loader = DataLoader(dev_dataset, batch_size=4, collate_fn=partial(collate_fn_nlpcc, name=name), shuffle=False)

best_valid_loss = float('inf')
best_f1 = 0.0
fgm = FGM(model)
for name, param in model.named_parameters():
    if param.requires_grad and 'embedding' in name:
        print(name)
for epoch in range(config.epoch):
    print('\n Epoch {:} / {:}'.format(epoch + 1, config.epoch))
    train_loss = train(model, train_data_loader, device, optimizer, fgm=fgm)
    dev_loss, dev_f1 = evaluate(model, dev_data_loader, device)
    #     if dev_loss<best_valid_loss:
    if dev_f1 > best_f1:
        best_f1 = dev_f1
        print('best f1 = ' + str(best_f1))
        best_valid_loss = dev_loss
        torch.save(model.state_dict(), model_dir+'/model_weights.pth')
    print('train loss {}'.format(train_loss))
    print('val loss {} val acc {}'.format(dev_loss, dev_f1))

del model 
import gc 
gc.collect()
torch.cuda.empty_cache()
model = BertClassifier(config)
model.to(device)
model.load_state_dict(torch.load(model_dir+'model_weights.pth'))

# %% [code] {"jupyter":{"outputs_hidden":false},"id":"bXs7vqLKRbVU","execution":{"iopub.status.busy":"2024-04-12T03:46:07.042530Z","iopub.execute_input":"2024-04-12T03:46:07.042825Z","iopub.status.idle":"2024-04-12T03:46:07.056379Z","shell.execute_reply.started":"2024-04-12T03:46:07.042793Z","shell.execute_reply":"2024-04-12T03:46:07.055521Z"}}
test_dataset = NLPCCTaskDataSet(filepath=config.test_file, mini_test=False, is_test=False)
test_data_loader = DataLoader(test_dataset, batch_size=8, collate_fn=partial(collate_fn_nlpcc, name=name),
                              shuffle=False)
print(len(test_dataset.dataset))
golds, pred_like = test(model, test_data_loader)
print(len(golds))
print(golds[:5])
print(pred_like[:5])

from sklearn.metrics import classification_report

print(classification_report(golds, [int(p) for p in pred_like]))
print(len(golds))
writer = open('test_en_pred_fold+'+args.fold+'.jsonl', 'a+', encoding='utf-8')
label_map = {0: 'NEG', 1: 'POS', 2: 'NEU'}
desc2label = {'NEG': 0, 'POS': 1, 'NEU': 2}
right = 0
total = 0
golds = []
preds = []
# writer = open('')
train_data = []
for pred, t in zip(pred_like, test_dataset.dataset):
    t['pred'] = pred + 1
    total += 1
    if int(t['pred']) == int(t['label']):
        right += 1
    #     train_data.append(t)
    writer.write(json.dumps(t, ensure_ascii=False) + '\n')
writer.close()
print(float(right) / total)
