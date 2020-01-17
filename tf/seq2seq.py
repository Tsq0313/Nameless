import tensorflow as tf
import tensorflow_hub as hub
from keras.utils.np_utils import to_categorical
from bert import bert_tokenization as tokenization
import pandas as pd
import numpy as np
import json
import os

max_seq_len = 128
n_hidden_unit = 128

# Refer to https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

with open("female2male_part1.txt", "r") as f:
  questions = f.readlines()
  questions = [q.lower().strip() for q in questions]

with open("female2male_part2.txt", "r") as f:
  answers = f.readlines()
  answers = [a.lower().strip() for a in answers]

outdir = "./"

# Shuffle
questions = np.array(questions)
answers = np.array(answers)
perm = np.random.permutation(len(answers))
shuffled_input = questions[perm]
shuffled_label = answers[perm]

def build_vocab_dict(tokenizer, questions, answers):
    vocab = {}
    for sample in zip(questions, answers):
        question = ["[CLS]"] + tokenizer.tokenize(sample[0]) + ["[SEP]"]
        answer = ["[CLS]"] + tokenizer.tokenize(sample[1]) + ["[SEP]"]

        for q in question:
            vocab[q] = len(vocab) if q not in vocab else vocab[q]
        for a in answer:
            vocab[a] = len(vocab) if a not in vocab else vocab[a]
    vocab["[PAD]"] = len(vocab)
    vocab_idx2word = {vocab[key] : key for key in vocab.keys()}

    return vocab, vocab_idx2word

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

def batch_iter(encode_idt, encode_mask, encode_seg, decode_idt, decode_mask, decode_seg, label, batch_size, max_seq_length, vocab_size):
    # assert len(idt) == len(mask) == len(seg) == len(label)
    data_size = len(encode_idt)
    iter_num = data_size // batch_size # Avoid dimension disagreement

    for i in range(iter_num):
        start_index = i * batch_size
        end_index = start_index + batch_size

        encode_ids = encode_idt[start_index: end_index]
        encode_masks = encode_mask[start_index: end_index]
        encode_segs = encode_seg[start_index: end_index]

        decode_ids = decode_idt[start_index: end_index]
        decode_masks = decode_mask[start_index: end_index]
        decode_segs = decode_seg[start_index: end_index]

        labels = label[start_index: end_index]
        print(encode_ids.shape)        
        yield [encode_ids, encode_masks, encode_segs, decode_ids, decode_masks, decode_segs], to_categorical(labels, vocab_size)

def prepare_data(questions, answers, vocab, tokenizer, max_seq_length):
    data = [["[CLS]"] + tokenizer.tokenize(line) + ["[SEP]"] for line in questions]
    label = [["[CLS]"] + tokenizer.tokenize(line) + ["[SEP]"] for line in answers]

    # Generate word length for LSTM model
    d_length = [len(d) for d in data]
    l_length = [len(l) for l in label]
    max_length = max([max(d_length), max(l_length)])

    # Expand training dataset based on max length of sentence
    for i in range(len(data)):
        length = len(data[i])
        if length < max_length:
            data[i] += ["[PAD]"] * (max_length - length)

    for i in range(len(label)):
        length = len(label[i])
        if length < max_length:
            label[i] += ["[PAD]"] * (max_length - length)

    encode_ids = [get_ids(token, tokenizer, max_seq_length) for token in data]
    encode_masks = [get_masks(token, max_seq_length) for token in data]
    encode_segs = [get_segments(token, max_seq_length) for token in data]

    decode_ids = [get_ids(token, tokenizer, max_seq_length) for token in label]
    decode_masks = [get_masks(token, max_seq_length) for token in label]
    decode_segs = [get_segments(token, max_seq_length) for token in label]

    labels = []
    for tag in label:
        tags = [vocab[idx] for idx in tag]
        labels.append(tags)
        
    return encode_ids, encode_masks, encode_segs, decode_ids, decode_masks, decode_segs, labels

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
if os.path.isfile('vocab_w2i.json') and os.path.isfile('vocab_i2w.json'):
    with open('vocab_w2i.json', 'r') as f:
        vocab_w2i = json.load(f)
    with open('vocab_i2w.json', 'r') as f:
        vocab_i2w = json.load(f)

else:
    vocab_w2i, vocab_i2w = build_vocab_dict(tokenizer, questions, answers)
    with open('vocab_w2i.json', 'w') as f:
        json.dump(vocab_w2i, f)
    with open('vocab_i2w.json', 'w') as f:
        json.dump(vocab_i2w, f)

vocab_size = len(vocab_i2w)

encode_ids, encode_masks, encode_segs, decode_ids, decode_masks, decode_segs, labels = prepare_data(shuffled_input, shuffled_label, vocab_w2i, tokenizer, max_seq_len)

encode_ids = np.array(encode_ids)
encode_masks = np.array(encode_masks)
encode_segs = np.array(encode_segs)
decode_ids = np.array(decode_ids)
decode_masks = np.array(decode_masks)
decode_segs = np.array(decode_segs)
labels = np.array(labels)

# Shuffle
perm = np.random.permutation(encode_ids.shape[0])
encode_ids = encode_ids[perm, :]
encode_masks = encode_masks[perm, :]
encode_segs = encode_segs[perm, :]
decode_ids = decode_ids[perm, :]
decode_masks = decode_masks[perm, :]
decode_segs = decode_segs[perm, :]
labels = labels[perm, :]

# Split dataset
size = len(shuffled_label)
unit = int(size / 5)

train_encode_ids = encode_ids[: unit * 4, :]
train_encode_masks = encode_masks[: unit * 4, :]
train_encode_segs = encode_segs[: unit * 4, :]
test_encode_ids = encode_ids[unit * 4 : , :]
test_encode_masks = encode_masks[unit * 4 : , :]
test_encode_segs = encode_segs[unit * 4 : , :]
train_decode_ids = decode_ids[: unit * 4, :]
train_decode_masks = decode_masks[: unit * 4, :]
train_decode_segs = decode_segs[: unit * 4, :]
test_decode_ids = decode_ids[unit * 4 : , :]
test_decode_masks = decode_masks[unit * 4 : , :]
test_decode_segs = decode_segs[unit * 4 : , :]

train_labels = labels[: unit * 4, 1 :]
test_labels = labels[unit * 4 : , 1 :]

def seq2seq_model(vocab_size, max_seq_length=128, n_hidden_unit=128):
  # Encoder
  encode_id = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="encode_input_word_ids")
  encode_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                      name="encode_input_mask")
  encode_seg_id = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, 
                                        name="encode_input_segment_ids")
  
  encode_input = [encode_id, encode_mask, encode_seg_id]
  _, encode_embedding = bert_layer(encode_input)
  encode_LSTM = tf.keras.layers.LSTM(n_hidden_unit, return_state=True)
  encode_output, state_h, state_c = encode_LSTM(encode_embedding)
  
  # Decoder
  decode_id = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="decode_input_word_ids")
  decode_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                      name="decode_input_mask")
  decode_seg_id = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, 
                                        name="decode_input_segment_ids")
  decode_input = [decode_id, decode_mask, decode_seg_id]
  _, decode_embedding = bert_layer(decode_input)
  decode_LSTM = tf.keras.layers.LSTM(n_hidden_unit, return_state=True, return_sequences=True)
  decode_output, _, _ = decode_LSTM(decode_embedding, initial_state=[state_h, state_c])
  
  output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))(decode_output[:, : -1, :])
  model = tf.keras.models.Model([encode_input, decode_input], output)
  
  return model

model = seq2seq_model(vocab_size=vocab_size, max_seq_length=max_seq_len, n_hidden_unit=n_hidden_unit)
model.summary()

# Train the model
batch_size = 8
steps_per_epoch = len(train_encode_ids) // batch_size
train_data = batch_iter(train_encode_ids, train_encode_masks, train_encode_segs, train_decode_ids, train_decode_masks, train_decode_segs, train_labels, batch_size, max_seq_len, vocab_size)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train_data = [train_encode_ids, train_encode_masks, train_encode_segs, train_decode_ids, train_decode_masks, train_decode_ids]
validation_data = [test_encode_ids, test_encode_masks, test_encode_segs, test_decode_ids, test_decode_masks, test_decode_ids]
model.fit_generator(train_data, epochs=20, verbose=1, steps_per_epoch=128, validation_data=(validation_data, to_categorical(test_labels, vocab_size)))
# model.fit(train_data, to_categorical(train_labels, vocab_size), epochs=5)
tf.keras.models.save_model(model, 'generate_model.h5')

model = tf.keras.models.load_model('generate_model.h5', custom_objects={'KerasLayer':hub.KerasLayer})
blank_answer = np.array([['[CLS]'] + ['[PAD]'] * 127] * len(test_encode_ids))

test_encode_ids = tf.convert_to_tensor(test_encode_ids, dtype=tf.int32, name='input_word_ids')
test_encode_masks = tf.convert_to_tensor(test_encode_masks, dtype=tf.int32, name='input_mask')
test_encode_segs = tf.convert_to_tensor(test_encode_segs, dtype=tf.int32, name='input_type_ids')

test_decode_ids = tf.convert_to_tensor(test_decode_ids, dtype=tf.int32, name='input_word_ids')
test_decode_masks = tf.convert_to_tensor(test_decode_masks, dtype=tf.int32, name='input_mask')
test_decode_segs = tf.convert_to_tensor(blank_answer, dtype=tf.int32, name='input_type_ids')

test_data = [test_encode_ids, test_encode_masks, test_encode_segs, test_decode_ids, test_decode_masks, test_decode_segs]
predictions = model.predict(test_data, verbose=1)

# print(len(vocab_i2w))
with open("output.txt", "w") as f:
    for p in predictions:
        p = np.argmax(p, axis=1)
        p = [str(pp) for pp in p] 
        f.write('\t'.join(p) + '\n')
