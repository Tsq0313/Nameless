import os
import numpy as np
import nltk
import torch
from time import time
import torchtext
from torchtext import data, datasets
from torchtext.vocab import GloVe
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from utils import read_data, divide_dataset,  ClassificationMetrics, data_analysis
from model import TextCNN, TextLSTM

from transformers import BertTokenizer, BertForSequenceClassification
# Load data
def make_data(fields):
    examples = []
    premise = ""
    with open("../standard_lines.txt", "r") as f:
        for line in f.readlines():
            line = line.split("\t")
            text = line[-2]
            label = line[-1]
            
            text = text.lower()
            text = text.replace("<i>","").replace("</i>","")
            text = text.replace("<u>","").replace("</u>","")
            text = text.replace("'bout", "about")
            text = text.replace("y'know", "you know")
            text = text.replace("goin'", "going")
            text = nltk.word_tokenize(text)
            l = len(text)+len(premise)
            if l>=4 and l < 256 and len(premise)>0 :
                example = data.Example.fromlist(
                    [premise+["[SEP]"]+text, label],
                    fields
                )
                examples.append(example)
            premise = text
    return data.Dataset(examples, fields)
def postproc(text, x):
    ret = []
    for sent in text:
        ret.append(tokenizer.encode(sent))
    return ret
#parameters
vocab_size = 20000
batch_size = 32
embed_size = 300
num_filters = 128
hidden_size = 300
n_epochs = 10
vocab_size = 20000
num_classes = 2
pretrain = True
use_bert = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##################
nltk.download("punkt")
if use_bert:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    TEXT  = data.Field(use_vocab=False, batch_first=True, lower=True, postprocessing=postproc, pad_token="[PAD]")
else:
    TEXT  = data.Field(lower=True, batch_first=True)#, preprocessing=preproc
LABEL = data.Field(sequential=False, batch_first=True, use_vocab=False)
fields = [
    #    ("premise", TEXT),
        ("text", TEXT),
        ("label", LABEL)
        ]
__train = make_data(fields)
print("make data finish")
train, valid = divide_dataset(__train, 0.9, fields)
TEXT.build_vocab(train, max_size = vocab_size)
if pretrain:
    TEXT.vocab.load_vectors(vectors=GloVe(name='6B', dim=300))
train_size = len(train)
valid_size = len(valid)
print("device: ",device)
print("train size :", train_size)
print("valid size :", valid_size)
train_iter, valid_iter = data.BucketIterator.splits(
            (train, valid), batch_size=batch_size, sort_key=lambda x:len(x.text), sort_within_batch=True,device=device)
data_analysis(train, TEXT)


if use_bert:
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes).to(device)
else:
    model = TextCNN (TEXT.vocab, embed_size, num_filters, num_classes, pretrain=pretrain).to(device)
  #  model = TextLSTM(TEXT.vocab, embed_size, hidden_size, num_classes, pretrain=pretrain).to(device)
 #   model = ESIM(TEXT.vocab, embed_size, hidden_size, num_classes, pretrain=pretrain).to(device)

print(model)
total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total:', total_num)
print('Trainable:', trainable_num)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

"""
print(batch.text)
print(batch.label)
print(output)
"""
def run_epoch(data_iter, train):
    # Train the model
    metrics = ClassificationMetrics(criterion)
    if train:
        model.train()
    else:
        model.eval()
    for i, batch in enumerate(data_iter):
      #  premise = batch.premise.to(device)
        text = batch.text.to(device)
        labels = batch.label.to(device)
        
        if train:
            output = model(text)
            if use_bert:
                output = output[0]
            loss = metrics.update(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i%1000==0:
                print(i)
    # Adjust the learning rate
   # scheduler.step()
        else:
            with torch.no_grad():
                output = model(text)
                if use_bert:
                    output = output[0]
                loss = metrics.update(output, labels)

    return metrics
print("traing...")
for epoch in range(n_epochs):

    start_time = time()
    train_metrics = run_epoch(train_iter, True)
    valid_metrics = run_epoch(valid_iter, False)

    secs = int(time() - start_time)
    print("epoch", epoch,"finished in "+str(secs)+"s")
    print("train:", train_metrics)
    print("valid:", valid_metrics)
