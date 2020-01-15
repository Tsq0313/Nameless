import random
from torchtext import data, datasets
from collections import Counter
def read_data(size_limit):
    f_chars_file = "F_text.txt"
    f_chars_id = set()
    with open(f_chars_file, "r") as f:
       for line in f.readlines():
           line = line.split(" +++$+++ ")
           k = line[0]
           f_chars_id.add(k)

    m_chars_file = "M_text.txt"
    m_chars_id = set()
    with open(m_chars_file, "r") as f:
        for line in f.readlines():
            line = line.split(" +++$+++ ")
            k = line[0]
            m_chars_id.add(k)

    f_lines = []
    m_lines = []
    lines_file = "lines.txt"
    with open(lines_file, "r") as f:
        for line in f.readlines():
            line = line.split(" +++$+++ ")
            k = line[1]
            v = line[4].strip()
            if len(v.split()) > 128: continue
            if len(v.split()) < 4: continue
            v = v.lower().replace("<t>","").replace("</t>","")
            v = v.replace("<u>","").replace("</u>","")
            v = v.replace("'bout", "about")
            v = v.replace("y'know", "you know")
            if k in f_chars_id:
                f_lines.append(v)
            elif k in m_chars_id:
                m_lines.append(v)

            if len(f_lines)+len( m_lines)>size_limit:
                break
    return m_lines, f_lines

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

def data_analysis(train, TEXT):
    for i in range(5):
        print(train[i].text, train[i].label)
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
    label_freqs  = Counter([example.label for example in train])
    vocab_freqs  = TEXT.vocab.freqs
    
    length_freqs = Counter([len(example.text) for example in train])
    print("context: min len :",min(length_freqs),"max len :",max(length_freqs))
    for i in range(1,6):
        print(i, length_freqs[i])
    
    print(label_freqs)
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

