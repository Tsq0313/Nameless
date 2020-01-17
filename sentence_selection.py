#dataset
import os
import torch
import random
import time
import math
from pytorch_transformers import DistilBertModel as BertModel
from pytorch_transformers import DistilBertTokenizer as BertTokenizer
import time
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.set_device(0)

SPL_SYMS = ['<PAD>','<BOS>', '<EOS>', '<UNK>']

#Data Reader
class STSCorpus(object):
  def __init__(self,
              file,
              vocab=None,
              cuda=False,
              batch_size=1, bert_format=0):
    self.bert_format = bert_format
    if self.bert_format == 0:
      self.bert_tokenizer = None
      self.max_vocab = 64000
    else:
      self.bert_tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
      self.max_vocab = self.bert_tokenizer.vocab_size
    self.max_size = 0
    self.batch_size = batch_size
    self.vocab = self.make_vocab(file, vocab)
    self.idx2vocab = self.make_idx2vocab(self.vocab)
    self.data = self.numberize(file, self.vocab, cuda)
    self.batch_data = self.batchify()
    self.data_size = len(self.batch_data)

  def batchify(self,):
    self.batch_data = []
    curr_batch = []
    max_x1, max_x2 = 0, 0
    for x1, x2, y in self.data:
      if len(curr_batch) < self.batch_size:
        curr_batch.append((x1, x2, y))
        max_x1 = max(max_x1, x1.shape[1])
        if self.bert_format == 0:
          max_x2 = max(max_x2, x2.shape[1]) 
      else:
        
        _x1, _x2, _y = zip(*curr_batch)
        
        
        if self.bert_format == 0:
          _x1 = [torch.cat((torch.zeros(1, max_x1 - i.shape[1]).type_as(i), i), dim=1) for i in _x1]
          batch_x1 = torch.cat(_x1, dim=0)
          _x2 = [torch.cat((torch.zeros(1, max_x2 - i.shape[1]).type_as(i), i), dim=1) for i in _x2]
          batch_x2 = torch.cat(_x2, dim=0) if _x2[0] is not None else None
        else:
          _x1 = [torch.cat((i, torch.zeros(1, max_x1 - i.shape[1]).type_as(i)), dim=1) for i in _x1]
          batch_x1 = torch.cat(_x1, dim=0)
          batch_x2 = None
        batch_y = torch.cat(_y, dim=0)
        self.batch_data.append((batch_x1, batch_x2, batch_y))
        curr_batch = []
        max_x1, max_x2 = 0, 0
    # remaining items in curr_batch
    if len(curr_batch) > 0:
      print(len(self.batch_data),  max_x1, max_x2)
      _x1, _x2, _y = zip(*curr_batch)
      
      
      if self.bert_format == 0:
        _x1 = [torch.cat((torch.zeros(1, max_x1 - i.shape[1]).type_as(i), i), dim=1) for i in _x1]
        batch_x1 = torch.cat(_x1, dim=0)
        _x2 = [torch.cat((torch.zeros(1, max_x2 - i.shape[1]).type_as(i), i), dim=1) for i in _x2]
        batch_x2 = torch.cat(_x2, dim=0) if _x2[0] is not None else None
      else:
        _x1 = [torch.cat((i, torch.zeros(1, max_x1 - i.shape[1]).type_as(i)), dim=1) for i in _x1]
        batch_x1 = torch.cat(_x1, dim=0)
        batch_x2 = None
      batch_y = torch.cat(_y, dim=0)
      self.batch_data.append((batch_x1, batch_x2, batch_y))
    return self.batch_data

  def numberize(self, txt, vocab, cuda=False):
    data = []
    max_size = 0
    with open(txt, 'r', encoding='utf8') as corpus:
      for l in corpus:
        l1, l2, y = l.split('\t')[-3:]
        #print(l)
        y = torch.Tensor([[float(y)]]).float()
        if self.bert_format == 0:
          d1 = [vocab['<BOS>']] + [vocab.get(t, vocab['<UNK>']) for t in l1.strip().split()] + [vocab['<EOS>']]
          d1 = torch.Tensor(d1).long()
          d1 = d1.unsqueeze(0) # shape = (1, N)
          d2 = [vocab['<BOS>']] + [vocab.get(t, vocab['<UNK>']) for t in l2.strip().split()] + [vocab['<EOS>']]
          d2 = torch.Tensor(d2).long()
          d2 = d2.unsqueeze(0) # shape = (1, N)
          max_size = max(d1.shape[1], d2.shape[1], max_size)
          if cuda:
            d1 = d1.cuda()
            d2 = d2.cuda()
            y = y.cuda()
        elif self.bert_format == 1:
          _d1 = torch.Tensor(self.bert_tokenizer.encode("[CLS] " + l1 + " [SEP]")).long()
          _d2 = torch.Tensor(self.bert_tokenizer.encode(" " + l2 + " [SEP]")).long()
          d = torch.cat([_d1, _d2], dim=0).unsqueeze(0)
          max_size = max(d.shape[1], max_size)
          if cuda:
            d1 = d.cuda()
            d2 = None
            y = y.cuda()
        else:
          pass
        data.append((d1, d2, y))
    self.max_size = max_size
    return data

  def make_idx2vocab(self, vocab):
    if vocab is not None:
      idx2vocab = {v: k for k, v in vocab.items()}
      return idx2vocab
    else:
      return None

  def make_vocab(self, txt, vocab):
    if vocab is None and txt is not None:
      vc = {}
      for line in open(txt, 'r', encoding='utf-8').readlines():
        #print("line:" + line)
        x1, x2, y = line.strip().split('\t')[-3:]
        for w in x1.split() + x2.split():
          vc[w] = vc.get(w, 0) + 1
      cv = sorted([(c, w) for w, c in vc.items()], reverse=True)
      cv = cv[:self.max_vocab]
      _, v = zip(*cv)
      v = SPL_SYMS + list(v)
      vocab = {w: idx for idx, w in enumerate(v)}
      return vocab
    else:
      return vocab

  def get(self, idx):
    return self.batch_data[idx]

class Classifier(torch.nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.1,
                 max_grad_norm=5.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.max_grad_norm = max_grad_norm
        self.dropout_layer = torch.nn.Dropout(p = dropout)
        
        if max(vocab_size,embedding_size ,hidden_size,num_layers) > 0:
          self.embedding_layer = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = self.embedding_size)
          
          self.uni_RNN_LSTM_layer = torch.nn.LSTM(input_size = self.embedding_size, hidden_size = self.hidden_size, num_layers=self.num_layers,  dropout = dropout, batch_first= True)
          self.output = torch.nn.Linear(in_features=self.hidden_size * 2, out_features = 1)
          


          self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
        else:
          pass
        self.loss = torch.nn.BCELoss(reduction='mean')
          

    def predict(self, x1, x2):
        """ Generates a prediction and probability for each input instance
        Args:
            x1: sequence of input tokens for the first sentence
            x2: sequence of input tokens for the second sentence
        Returns:
            out: sequence of output predictions (probabilities) for each instance
            pred: the discrete prediction from the output probabilities
        """
        batch_size, seq_len = x1.shape
        batch_size2, seq_len2 = x2.shape
        assert batch_size == batch_size2
        
        emb_x1 = self.dropout_layer(self.embedding_layer(x1))
        
        emb_x2 = self.dropout_layer(self.embedding_layer(x2))

        h, c = (torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())
        x1_out, (x1_hidden, x1_cell) = self.uni_RNN_LSTM_layer(emb_x1, (h, c))
        x2_out, (x2_hidden, x2_cell) = self.uni_RNN_LSTM_layer(emb_x2, (h, c))
        final_hidden = torch.cat((x1_out[:,-1,:].squeeze(1), x2_out[:,-1,:].squeeze(1)), -1)
        final_hidden = self.dropout_layer(final_hidden)
        out = torch.sigmoid(self.output(final_hidden))

        pred = out.clone().detach()
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return out, pred

    def forward(self, x1, x2, y):
        out, pred = self.predict(x1,x2)
        loss = self.loss(out, y)

        assert pred.shape == y.shape
        acc = (pred == y).sum().item() / y.numel()
        return loss, acc

    def train_step(self, x1, x2, y):
        self.optimizer.zero_grad()
        _loss, acc = self(x1, x2, y) # calls self.forward(x, y)
        _loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.parameters()),
                                                   self.max_grad_norm)

        if math.isnan(grad_norm):
            print('skipping update grad_norm is nan!')
        else:
            self.optimizer.step()
        loss = _loss.item()
        return loss, acc


class BERTClassifier(Classifier):
    def __init__(self,
                 dropout=0.1,
                 max_grad_norm=5.0):
        super().__init__(0, 0, 0, 0, dropout, max_grad_norm)
        self.output = torch.nn.Linear(768, 1)
        weight = torch.nn.init.normal_(torch.zeros(1,768), mean = 0, std = 0.05)
        self.output.weight = torch.nn.Parameter(weight)
        self.bert_model = BertModel.from_pretrained('distilbert-base-uncased')
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-5)

    def predict(self, x1, x2=None):
        assert x2 is None
        x2 = self.bert_model(x1)
        out = torch.sigmoid(self.output(x2[0][:,-1,:].squeeze(1)))

        pred = out.clone().detach()
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return out, pred

#Training Routine
def train(model, train_cropus, dev_corpus, max_epochs):
  sum_loss, sum_acc = 0., 0.
  train_instances_idxs = list(range(train_corpus.data_size))
  st = time.time()
  for epoch_i in range(max_epochs):
    sum_loss, sum_acc = 0., 0.
    random.shuffle(train_instances_idxs)
    model.train()
    for i in train_instances_idxs:
      x1, x2, y = train_corpus.get(i)
      l, a = model.train_step(x1, x2, y)
      sum_loss += l
      sum_acc += a
    print(f"epoch: {epoch_i} time elapsed: {time.time() - st:.2f}")
    print(f"train loss: {sum_loss/train_corpus.data_size:.4f} train acc: {sum_acc/train_corpus.data_size:.4f}")
    sum_loss, sum_acc = 0., 0.
    model.eval()
    for dev_i in range(dev_corpus.data_size):
      x1, x2, y = dev_corpus.get(dev_i)
      with torch.no_grad():
        l, a = model(x1, x2, y)
        sum_loss += l
        sum_acc += a
    print(f"  dev loss: {sum_loss/dev_corpus.data_size:.4f}   dev acc: {sum_acc/dev_corpus.data_size:.4f}")
  return model

# Evaluation Routine
def evaluate(model, test_corpus):
  print('Predictions:')
  sum_acc = 0.0
  model.eval()
  for test_i in range(test_corpus.data_size):
    x1, x2, y = test_corpus.get(test_i)
    _, pred = model.predict(x1, x2)
    sum_acc += (1 if pred.item() == y.item() else 0)
  print(f"Avg acc: {sum_acc/test_corpus.data_size:.4f}")
  
def find_best_pair(model, test_corpus, input_sentence):
      print("The best answer is:")
      biggest_prob = 0
      answer_index = 0
      model.eval()
      for test_i in range(test_corpus.data_size):
        x1, x2, y = test_corpus.get(test_i)
        out,  pred = model.predict(input, x2)
        out = out.clone().detach()
        if out.item() > biggest_prob:
          biggest_prob = out.item()
          answer_index = test_i
      x1, x2, y = test_corpus.get(test_i)
      return x2

if __name__ == '__main__':
  #Creating train, dev and test data objects. (with bert_format=1) and places the data on the GPU.
  train_corpus = STSCorpus(file='train.tsv',
                            cuda=True,
                            batch_size=32, bert_format=1)
  dev_corpus = STSCorpus(file='dev.tsv', vocab=train_corpus.vocab,
                          cuda=True,
                          batch_size=32,bert_format=1)
  test_corpus = STSCorpus(file='test.tsv', vocab=train_corpus.vocab,
                          cuda=True,
                          batch_size=1,bert_format=1)
  print(train_corpus.data_size, dev_corpus.data_size, test_corpus.data_size)


  bert_model = BERTClassifier()
  bert_model = bert_model.cuda()
  print(bert_model, '\ncontains', sum([p.numel() for p in bert_model.parameters() if p.requires_grad]), 'parameters')

  #bert_model = train(bert_model, train_corpus, dev_corpus, 3) # takes ~1 hour
  bert_model = torch.load('2020-01-16-07:02:18fine-tuned-bert-model.pth.tar')
  evaluate(bert_model, test_corpus)

  input = 'I have to be up early.'
  answer = find_best_pair(bert_model, test_corpus, input)
  print(answer)

  '''
  time_now = int(time.time())
  time_local = time.localtime(time_now)
  dt = time.strftime("%Y-%m-%d-%H:%M:%S",time_local)
  torch.save(bert_model, dt + 'fine-tuned-bert-model.pth.tar')
  '''