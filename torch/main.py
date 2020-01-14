import os
import numpy as np
import nltk
import torch
import re
from time import time
from collections import Counter
import torchtext
from torchtext import data, datasets
from torchtext.vocab import GloVe
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from utils import read_data, divide_dataset,  ClassificationMetrics
from model import TextCNN, TextLSTM
# Load data
def make_data(fields):
    m_lines, f_lines = read_data(200)
    # Load Pre-Trained BERT Model via TF 2.0
    lines = m_lines + f_lines
    size = len(lines)
    print(len(m_lines), len(f_lines))
    max_seq_length = 128
    examples = []
    for text in m_lines:
        example = data.Example.fromlist(
            [text, 0],
            fields
        )
        examples.append(example)
    for text in f_lines:
        example = data.Example.fromlist(
            [text, 1],
            fields
        )
        examples.append(example)
    return data.Dataset(examples, fields)
#yparameters
vocab_size = 20000
batch_size = 32
embed_size = 300
num_filters = 128
hidden_size = 300
n_epochs = 10
vocab_size = 20000
num_classes = 2
pretrain = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##################
nltk.download("punkt")
TEXT  = data.Field(lower=True, batch_first=True, tokenize=nltk.word_tokenize)#, preprocessing=preproc
LABEL = data.Field(sequential=False, batch_first=True, use_vocab=False)
fields = [
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
            (train, valid), batch_size=batch_size, sort_key=lambda x:len(x.text), device=device)
#data_analysis()


#model = TextCNN (TEXT.vocab, embed_size, num_filters, num_classes, pretrain=pretrain).to(device)
model = TextLSTM(TEXT.vocab, embed_size, hidden_size, num_classes, pretrain=pretrain).to(device)
print(model)
total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total:', total_num)
print('Trainable:', trainable_num)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

batch = next(iter(train_iter))
output = model(batch.text)
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
        text = batch.text.to(device)
        labels = batch.label.to(device)
        
        if train:
            output = model(text)
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
