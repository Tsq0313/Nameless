import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Set
from collections import Counter
from time import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
class TextCNN(nn.Module):
    def __init__(self, vocab, embeding_size, filters, num_classes, pretrain = False):
        super(TextCNN, self).__init__()
        padding_idx = vocab.stoi["<pad>"]
        unknown_idx = vocab.stoi["<unk>"]
        vocab_size  = len(vocab.itos)
        self.embedding = nn.Embedding(vocab_size, embeding_size, padding_idx=padding_idx)
        if pretrain: 
            self.embedding.weight.data.copy_(vocab.vectors)
        self.conv1 = nn.Conv1d(embeding_size, filters, 2)
        self.conv2 = nn.Conv1d(embeding_size, filters, 3)
        self.conv3 = nn.Conv1d(embeding_size, filters, 4)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(filters*3, num_classes)
    def forward(self, x):
        """
            x : Tensor(B, L)
        """
        x = self.embedding(x)
     #   print(x.shape)
        x = x.transpose(2, 1)
        c1 = self.maxpool(self.conv1(x))
        c2 = self.maxpool(self.conv2(x))
        c3 = self.maxpool(self.conv3(x))
        c = torch.cat((c1,c2,c3), 1)
      #  print(c.shape)
        c = c.squeeze(2)
        c = self.fc(c)
       # print(c.shape)
        return c

class TextLSTM(nn.Module):
    def __init__(self, vocab, embeding_size, hidden_size, num_classes, pretrain = False):
        super().__init__()
        self.padding_idx = vocab.stoi["<pad>"]
        unknown_idx = vocab.stoi["<unk>"]
        vocab_size  = len(vocab.itos)
        self.embedding = nn.Embedding(vocab_size, embeding_size, padding_idx=self.padding_idx)
        if pretrain: 
            self.embedding.weight.data.copy_(vocab.vectors)

        self.LSTM = nn.LSTM(embeding_size, hidden_size, 1, batch_first = True, bidirectional=True)
        self.fc   = nn.Linear(hidden_size*6, num_classes)
        self.vec  = nn.Parameter(torch.zeros(1,hidden_size*2, 1))
        torch.nn.init.normal_(self.vec)

    def forward(self, x):
        """
           x : Tensor(B, L)
        """
        B = x.shape[0]
        x_len = [sum(x[i] != self.padding_idx) for i in range(B)]

        x = self.embedding(x)
      #  x = self.dropout(x)
        x = pack_padded_sequence(x, x_len, batch_first=True)
        x, _ = self.LSTM(x)
        x =  pad_packed_sequence(x, batch_first=True)
        x = x[0]

        y = self.vec.repeat(B,1,1)
        e = torch.bmm(x, y) # (B,L,1)
        mask = torch.ones_like(e)
        for i in range(B):
            mask[i,:x_len[i]] = 0
        e.data.masked_fill_(mask.bool(), -1e30)
        e = F.softmax(e, dim=1)
        att = torch.bmm(e.transpose(1,2), x).squeeze(1)

        x_m = x.transpose(1,2)
        x_max = torch.max_pool1d(x_m, x_m.shape[-1]).squeeze(-1)
        x_avg = torch.avg_pool1d(x_m, x_m.shape[-1]).squeeze(-1)

        out = torch.cat((x_max, x_avg, att), 1) #(B,6H)
        out = self.fc(out)
        return out

class ESIM(nn.Module):
    def __init__(self, vocab, embeding_size, hidden_size, num_classes, dropout = 0, pretrain = False):
        super(ESIM, self).__init__()
        self.padding_idx = vocab.stoi["<pad>"]
        self.hidden_size = hidden_size
        unknown_idx = vocab.stoi["<unk>"]
        vocab_size  = len(vocab.stoi)
        self.embedding = nn.Embedding(vocab_size, embeding_size, padding_idx=self.padding_idx)
        if pretrain:
            self.embedding.weight.data.copy_(vocab.vectors)
        self.LSTM = nn.LSTM(embeding_size, hidden_size, 1, batch_first = True, bidirectional=True)
        self.fc = nn.Linear(32*hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, y):
        """
            x : Tensor(B,L1)
            y : Tensor(B,L2)
            x_len : list[int]
            y_len : list[int]
            mask : Tensor(B,L1,L2) 
        """
        B = x.shape[0]
        x, x_len = self.step(x)
        y, y_len = self.step(y)
        y_T = y.transpose(1, 2)
        e = torch.bmm(x, y_T)
        mask = torch.ones_like(e)

        for i in range(B):
            mask[i,:x_len[i],:y_len[i]] = 0
    #    print(x_len)
     #   print(y_len)
      #  print(e[0])
        e.data.masked_fill_(mask.bool(), float("-inf"))
       # print(e[0])
        e_x = F.softmax(e, dim=1)
        e_y = F.softmax(e, dim=2)
        e_x.data.masked_fill_(mask.bool(), 0.0)
        e_y.data.masked_fill_(mask.bool(), 0.0)
        #print(e_x)
        x_att = torch.bmm(e_y, y) # (B, L1, 2H)
        y_att = torch.bmm(e_x.transpose(1,2), x)

        x_m = torch.cat((x, x_att, x-x_att, x*x_att), 2) #(B, L1, 8H)
        y_m = torch.cat((y, y_att, y-y_att, y*y_att), 2)
        x_m = x_m.transpose(1,2)
        y_m = y_m.transpose(1,2)
        x_m = self.dropout(x_m)
        y_m = self.dropout(y_m)

        x_max = torch.max_pool1d(x_m, x_m.shape[-1]).squeeze(-1) # (B, 8H)
        x_avg = torch.avg_pool1d(x_m, x_m.shape[-1]).squeeze(-1)
        y_max = torch.max_pool1d(y_m, y_m.shape[-1]).squeeze(-1)
        y_avg = torch.avg_pool1d(y_m, y_m.shape[-1]).squeeze(-1)
        out = torch.cat((x_max, x_avg, y_max, y_avg), 1) #(B,32H)
     #   out = torch.cat((y_max, y_avg), 1) #(B,32H)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    def step(self, x):
        """
            x : Tensor(B,L)
        return
            x : Tensor(B,L,2H)
        """
        B = x.shape[0]
        x_len = [sum(x[i] != self.padding_idx) for i in range(B)]
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.LSTM(x)
        return x, x_len
