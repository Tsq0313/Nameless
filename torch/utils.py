import random
from torchtext import data, datasets
from collections import Counter
import torch
def divide_dataset(__train, factor, fields):
    train_size = int(len(__train) * factor)
    valid_size = len(__train) - train_size
    valid_indices = list(range(train_size+valid_size))
    random.shuffle(valid_indices)
    train_indices = valid_indices[:train_size]
    valid_indices = valid_indices[train_size:]
    train = [__train[idx] for idx in train_indices]
    valid = [__train[idx] for idx in valid_indices]
    train = data.Dataset(train, fields)
    valid = data.Dataset(valid, fields)
    return train, valid

def data_analysis(train, valid, TEXT):
    ##check word vector
    v = TEXT.vocab.vectors
    vv= v.sum(1)
    l = [i for i in range(vv.shape[0]) if vv[i]==0]
    print("no vector: ", len(l))
    for i in l[0:20]:
        print(i,TEXT.vocab.itos[i])
    
    
    ##check voacb
    #data check
    for i in range(10):
        print(train[i].text,train[i].label)
    #data analysis
    vocab_freqs  = TEXT.vocab.freqs
    
    length_freqs = Counter([len(example.text) for example in train])
    print("context: min len :",min(length_freqs),"max len :",max(length_freqs))
    d = [0]*500
    for i in range(500):
        d[i] = length_freqs[i]
    print(d)
 #   label_freqs  = Counter([example.label for example in train])
    label_freqs  = Counter([l for example in train for l in example.label])
    print("train: ", label_freqs)
  #  label_freqs  = Counter([example.label for example in valid])
    label_freqs  = Counter([l for example in valid for l in example.label])
    print("valid: ", label_freqs)
    
    """
        词汇覆盖量分析
    """
    print(len(vocab_freqs))
    for i in [26,50,100,1000,10000, 30000, 40000, 60000, 80000]:
        print(i,vocab_freqs.most_common(i)[-1])
class ClassificationMetrics(object):
    def __init__(self, criterion):
        self.loss = 0
        self.size = 0
        self.acc = 0
        self.one_cnt = 0
        self.criterion = criterion
    def update(self, logit, label):
        size = logit.shape[0]
        loss = self.criterion(logit, label)
        self.loss += loss.item()*size
        self.size += size
        self.one_cnt += logit.argmax(1).sum().item()
        self.acc += (logit.argmax(1) == label).sum().item()
        return loss
    def __getitem__(self, key):
        return getattr(self, key)/self.size
    def __str__(self):
        return ("loss : %.2f, acc : %.2f, 1cnt : %.2f" %(self["loss"], self["acc"], self["one_cnt"]))

class MultiClassificationMetrics(object):
    def __init__(self, criterion):
        self.loss = 0
        self.size = 0
        self.acc = 0
        self.criterion = criterion
    def update(self, logit, label):
        label = label.to(torch.float32)
        size = logit.shape[0]
        loss = self.criterion(logit, label)
        self.loss += loss.item()*size
        self.size += size
        self.acc += ((logit>0) == label).sum().item()/10
        return loss
    def __getitem__(self, key):
        return getattr(self, key)/self.size
    def __str__(self):
        return ("loss : %.2f, acc : %.2f" %(self["loss"], self["acc"]))

