class ClassificationMetrics(object):
    def __init__(self, criterion):
        self.loss = 0
        self.size = 0
        self.acc = 0
        self.criterion = criterion
    def update(self, logit, label):
        size = logit.shape[0]
        loss = self.criterion(logit, label)
        self.loss += loss.item()
        self.size += size
        self.acc += (logit.argmax(1) == label).sum().item()
        return loss
    def __getitem__(self, key):
        return getattr(self, key)/self.size
    def __str__(self):
        return ("loss : %.2f, acc : %.2f" %(self["loss"], self["acc"]))

