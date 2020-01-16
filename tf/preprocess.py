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
f_chars_file = "./data/F_text.txt"
f_chars_id = []
with open(f_chars_file, "r") as f:
    for line in f.readlines():
        line = line.split(" +++$+++ ")
        k = line[0]
        f_chars_id.append(k)

m_chars_file = "./data/M_text.txt"
m_chars_id = []
with open(m_chars_file, "r") as f:
    for line in f.readlines():
        line = line.split(" +++$+++ ")
        k = line[0]
        m_chars_id.append(k)

f_lines = []
m_lines = []
lines_file = "./data/lines.txt"

with open(lines_file, "r") as f:
    for line in f.readlines():
        line = line.split(" +++$+++ ")
        k = line[1]
        v = line[4].strip()
        if k in f_chars_id:
            f_lines.append(v.lower())
        elif k in m_chars_id:
            m_lines.append(v.lower())

max_seq_length = 128
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

def batch_iter(id, mask, seg, label, batch_size, max_seq_length):
    """
        A mini-batch iterator to generate mini-batches for training neural network
        param data: a list of sentences. each sentence is a vector of integers
        param label: a list of labels
        param batch_size: the size of mini-batch
        param num_epochs: number of epochs
        return: a mini-batch iterator
        """
    assert len(id) == len(mask) == len(seg) == len(label)
    data_size = len(id)
    iter_num = data_size // batch_size # Avoid dimension disagreement

    for i in range(iter_num):
        start_index = i * batch_size
        end_index = start_index + batch_size

        ids = id[start_index: end_index]
        masks = mask[start_index: end_index]
        segs = seg[start_index: end_index]
        labels = label[start_index: end_index]
        #  print(len(labels))
            
        permutation = np.random.permutation(labels.shape[0])
        yield ids, masks, segs, labels

train_data_seq = []
train_data_word = []
train_label = []

def data_filter(m_lines, f_lines, total_size, tokenizer, max_seq_length):
    size = total_size // 2
    size = min([len(m_lines), len(f_lines), size])
    
    m_lines = [["[CLS]"] + tokenizer.tokenize(line) + ["[SEP]"] for line in m_lines]
    m_lines = [line for line in m_lines if len(line) < 128 and len(line) > 2]
    random.shuffle(m_lines)
    f_lines = [["[CLS]"] + tokenizer.tokenize(line) + ["[SEP]"] for line in f_lines]
    f_lines = [line for line in f_lines if len(line) < 128 and len(line) > 2]
    random.shuffle(f_lines)

    m_lines = m_lines[: size]
    f_lines = f_lines[: size]
    
    lines = m_lines + f_lines
    label = np.append(np.zeros(len(m_lines)), np.ones(len(f_lines)))
    
    ids = [get_ids(token, tokenizer, max_seq_length) for token in lines]
    masks = [get_masks(token, max_seq_length) for token in lines]
    segs = [get_segments(token, max_seq_length) for token in lines]
    
    # Shuffle
    perm = np.random.permutation(len(lines))
    ids = np.array(ids)
    masks = np.array(masks)
    segs = np.array(segs)
    ids = ids[perm, :]
    masks = masks[perm, :]
    segs = segs[perm, :]
    label = label[perm]
    
    return ids, masks, segs, label
    
# Parameters
total_size = 8000
batch_size = 128

ids, masks, segs, labels = data_filter(m_lines, f_lines, total_size, tokenizer, max_seq_length)
train_data = batch_iter(ids, masks, segs, labels, batch_size, max_seq_length)

for i, train_input in enumerate(train_data):
    print("No.", i, "iteration")
    input_id, input_mask, input_seg, label = train_input
    seq_data, word_data = model.predict([input_id, input_mask, input_seg])
    train_data_seq.extend(seq_data)
    train_data_word.extend(word_data)
    train_label.extend(label)
    print(len(train_label))
    print(len(train_data_seq), len(train_data_seq[0]))
    print(len(train_data_word), len(train_data_word[0]))
    
train_data_seq = np.array(train_data_seq)
train_data_word = np.array(train_data_word)
train_label = np.array(train_label)
np.save("data_seq.npy", train_data_seq)
np.save("data_word.npy", train_data_word)
np.save("label.npy", train_label)
