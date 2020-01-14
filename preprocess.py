import tensorflow as tf
from bert import bert_tokenization as tokenization
import os
import random
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import numpy as np
import re

# Load data
f_chars_file = "F_text.txt"
f_chars_id = []
with open(f_chars_file, "r") as f:
    for line in f.readlines():
        line = line.split(" +++$+++ ")
        k = line[0]
        f_chars_id.append(k)

m_chars_file = "M_text.txt"
m_chars_id = []
with open(m_chars_file, "r") as f:
    for line in f.readlines():
        line = line.split(" +++$+++ ")
        k = line[0]
        m_chars_id.append(k)

f_lines = []
m_lines = []
lines_file = "lines.txt"

with open(lines_file, "r") as f:
    for line in f.readlines():
        line = line.split(" +++$+++ ")
        k = line[1]
        v = line[4].strip()
        if k in f_chars_id:
            f_lines.append(v.lower())
        elif k in m_chars_id:
            m_lines.append(v.lower())

# Load Pre-Trained BERT Model via TF 2.0
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1",
                            trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids

def batch_iter(data, label, batch_size, num_epochs, tokenizer, max_seq_length):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    param data: a list of sentences. each sentence is a vector of integers
    param label: a list of labels
    param batch_size: the size of mini-batch
    param num_epochs: number of epochs
    return: a mini-batch iterator
    """
    assert len(data) == len(label) 
    data_size = len(data)
    epoch_length = data_size // batch_size # Avoid dimension disagreement

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            pre_data = data[start_index: end_index]
            input_id = []
            input_mask = []
            input_seg = []
            for line in pre_data:
                token = tokenizer.tokenize(line)
                token = ["[CLS]"] + token + ["[SEP]"]
                input_id.append(get_ids(token, tokenizer, max_seq_length))
                input_mask.append(get_masks(token, max_seq_length))
                input_seg.append(get_segements(token, max_seq_length))
                
            labels = label[start_index: end_index] 
            
            permutation = np.random.permutation(pre_data.shape[0])
            input_id = np.array(input_id)
            ipnut_id = input_id[permutation, :]
            input_mask = np.array(input_mask)
            ipnut_mask = input_mask[permutation, :]
            input_seg = np.array(input_seg)
            ipnut_seg = input_seg[permutation, :]
            labels = labels[permutation]
            
            yield input_id, ipnut_mask, input_seg, labels

train_data_seq = []
train_data_word = []
train_label = []

def data_filter(m_lines, f_lines, total_size):
    size = total_size // 2
    size = min([len(m_lines), len(f_lines), size])
    
    m_lines = [line for line in m_lines if line.split() < 121 and line.split() > 2]
    f_lines = [line for line in f_lines if line.split() < 121 and line.split() > 2]
    m_lines = random.shuffle(m_lines)
    f_lines = random.shuffle(f_lines)
    m_lines = m_lines[: size]
    f_lines = f_lines[: size]
    
    lines = m_lines + f_lines
    lines = np.array(lines)
    label = np.append(np.zeros(len(m_lines)), np.ones(len(f_lines)))
    
    # Shuffle
    perm = np.random.permutation(pre_data.shape[0])
    lines = lines[:, perm]
    label = label[:, perm]
    
    return lines, label          
    
# Parameters
total_size = 100000
batch_size = 128
num_epochs = 100000 // batch_size
size = len(lines)
max_seq_length = 128

lines, labels = data_filter(m_lines, f_lines, total_size)
train_data = batch_iter(lines, labels, batch_size, num_epochs, tokenizer, max_seq_length)

for train_input in train_data:
    input_id, input_mask, input_seg, label = train_input
    seq_data, word_data = model.predict([input_id, input_mask, input_segment])
    train_data_seq.append(train_data_seq, seq_data)
    train_data_word.append(train_data_word, word_data)
    train_label.append(label)
    print(train_data_seq.shape)
    print(train_data_word.shape)
    
train_data_seq = np.array(train_data_seq)
train_data_word = np.array(train_data_word)
train_label = np.array(train_label)
np.save("data_seq.npy", train_data_seq)
np.save("data_word.npy", train_data_word)
np.save("label.npy", train_label)
