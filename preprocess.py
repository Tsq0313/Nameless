import tensorflow as tf
from bert import bert_tokenization as tokenization
import os
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
        if len(v.split()) > 128:
            continue
        if k in f_chars_id:
            f_lines.append(v.lower())
        elif k in m_chars_id:
            m_lines.append(v.lower())

# Load Pre-Trained BERT Model via TF 2.0
max_seq_length = 128  # Your choice here.
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

# Prepare Data
lines = m_lines + f_lines
size = len(lines)
max_seq_length = 128

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
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
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

train_data = np.array([])
train_label = np.array([])

for i in range(size):
    line = lines[i]
    if line in f_lines:
        train_label = np.append(train_label, [1])
    elif line in m_lines:
        train_label = np.append(train_label, [0])
        
    token = tokenizer.tokenize(line)
    token = ["[CLS]"] + token + ["[SEP]"]
                
    input_id = get_ids(tokens, tokenizer, max_seq_length)
    input_mask = get_masks(stokens, max_seq_length)
    input_segment = get_segments(stokens, max_seq_length)

    _, data = model.predict([[input_ids],[input_masks],[input_segments]])
    train_data = np.append(train_data, data)

train_dataset = np.array(train_data)
train_labels = np.array(train_label)
np.save("data.npy", train_dataset)
np.save("label.npy", train_labels)
