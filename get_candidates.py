import os

train_lines = open('train.tsv','r').readlines()
candidate_file = open('candidate.txt','w')
for i in range(0, 2000):
    line = train_lines[i].strip()
    candidate = line.split('\t')[1].strip()
    candidate_file.write(candidate + '\n')
candidate_file.close()
