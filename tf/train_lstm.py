import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import pickle as pkl
import time
import json
import os
import sys
import csv

# Import word and sentence embeddings
word_input = np.load('data_word.npy')
label = np.load('label.npy')

print(word_input.shape, label.shape)
outdir = "/home/siqit4/Nameless/model"
# Generate word length for LSTM model
word_length = []
for i in range(len(word_input)):
    word_length = np.append(word_length, len(word_input[i]))
word_length = np.array(word_length)
max_length = max(word_length)

# Expand training dataset based on max length of sentence
tmp = [0] * 1024
for i in range(len(word_input)):
    length = len(word_input[i])
    if length < max_length:
        tmp_ = np.tile(tmp, (max_length - length, 1))
        word_input[i] = np.append(word_input[i], tmp_, axis=0)

# Shuffle
permutation = np.random.permutation(label.shape[0])
# np.save('shuffled order', permutation)
# permutation = np.load('shuffled order.npy')
shuffled_word_input = word_input[permutation, :, :]
shuffled_label = label[permutation]
shuffled_length = word_length[permutation]

# Split dataset
size = len(shuffled_label)
unit = int(size / 5)

print(unit)
print(word_input.shape, label.shape, word_length.shape)
print(shuffled_word_input.shape, shuffled_label.shape, shuffled_length.shape)
train_data = shuffled_word_input[: unit * 4]
train_label = shuffled_label[: unit * 4]
train_length = shuffled_length[: unit * 4]
test_data = shuffled_word_input[unit * 4 : ]
test_label = shuffled_label[unit * 4 :]
test_length = shuffled_length[unit * 4 :]

print(train_data.shape, train_label.shape, train_length.shape) 
# Mini-batch
def batch_iter(data, label, length, batch_size, num_epochs):
    """
        A mini-batch iterator to generate mini-batches for training neural network
        param data: a list of sentences. each sentence is a vector of integers
        param labels: a list of labels
        param batch_size: the size of mini-batch
        param num_epochs: number of epochs
        return: a mini-batch iterator
        """
    print(len(data), len(label), len(length))
    assert len(data) == len(label) == len(length)
    data_size = len(data)
    epoch_length = data_size // batch_size # Avoid dimension disagreement
    
    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size
            
            xdata = data[start_index: end_index]
            ydata = label[start_index: end_index]
            sequence_length = length[start_index: end_index]
            
            permutation = np.random.permutation(xdata.shape[0])
            xdata = xdata[permutation, :, :]
            ydata = ydata[permutation]
            sequence_length = sequence_length[permutation]
            
            yield xdata, ydata, sequence_length

# Build LSTM model
class LSTM_Model(object):
    def __init__(self):
        self.hidden_size = 1024
        self.num_layers = 1
        self.l2_reg_lambda = 0.001
        
        # Placeholders
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, 128, 1024], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.seq_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')
        # self.rate = tf.placeholder(dtype=tf.float32, shape=[None], name='rate')
        # L2 loss
        self.l2_loss = tf.constant(0.0)
        
        # Word embedding
        with tf.name_scope('embedding'):
            inputs = self.input_x
        
        # Input dropout
        self.inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)
        self.final_state = self.lstm()
        
        # Softmax output layer
        with tf.name_scope('sigmoid'):
            # softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size, self.num_classes], dtype=tf.float32)
            sigmoid_w = tf.get_variable('sigmoid_w', shape=[self.hidden_size, 1], dtype=tf.float32)
            sigmoid_b = tf.get_variable('sigmoid_b', shape=[1], dtype=tf.float32)
            
            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(sigmoid_w)
            self.l2_loss += tf.nn.l2_loss(sigmoid_b)
            
            self.predictions = tf.matmul(self.final_state, sigmoid_w) + sigmoid_b
            self.predictions = tf.squeeze(self.predictions)
            self.predictions = tf.nn.sigmoid(self.predictions, name='predictions')
        
        
        # Loss (MSE)
        with tf.name_scope('loss'):
            tvars = tf.trainable_variables()
            
            # L2 regularization for LSTM weights
            for tv in tvars:
                if 'kernel' in tv.name:
                    self.l2_loss += tf.nn.l2_loss(tv)
        
            losses = tf.square(self.predictions - self.input_y)
            self.cost = tf.reduce_mean(losses, name="loss") + self.l2_reg_lambda * self.l2_loss
        
        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(tf.round(self.predictions), self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')


    def lstm(self):
    
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                   forget_bias= 1.0,
                                   state_is_tuple=True,
                                   reuse=tf.get_variable_scope().reuse)
        
        
        # Add dropout to cell output
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        
        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell], state_is_tuple=True)
        
        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        
        # Dynamic LSTM
        with tf.variable_scope('LSTM'):
            _, state = tf.nn.dynamic_rnn(cell,
                                         inputs=self.inputs,
                                         initial_state=self._initial_state,
                                         sequence_length=self.seq_length)
        
        output = state[self.num_layers - 1].h
        
        return output

# Model hyperparameters (LSTMs are all single layer)
hidden_size = 1024 # Number of hidden units in the LSTM cell
keep_prob = 0.5 # Dropout keep probability
learning_rate = 1e-5 # 注意 BERT 在训练时需要调小训练率
l2_reg_lambda = 0.001 # L2 regularization lambda

# Training parameters
batch_size = 128
num_epoch = 100 # 100 epochs 后如果没变化则停止，否则接着训练 100 epochs （手动观察）
decay_rate = 1
decay_steps = 100000 # Learning rate decay rate. Range: (0, 1]
save_every_steps = 100
evaluate_every_steps = 10 # Evaluate the model on validation set after this many steps
num_checkpoint = 50 # number of models to store

# Train
# =============================================================================
train_data = batch_iter(train_data, train_label, train_length, batch_size, num_epoch)
with tf.Graph().as_default():
    with tf.Session() as sess:
        classifier = LSTM_Model()
        # Train procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Learning rate decay
        starter_learning_rate = learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   decay_steps,
                                                   decay_rate,
                                                   staircase=True)
                                                   
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(classifier.cost)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        # Summaries
        loss_summary = tf.summary.scalar('Loss', classifier.cost)
        accuracy_summary = tf.summary.scalar('Accuracy', classifier.accuracy)
        
        # Train summary
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(outdir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        # Validation summary
        valid_summary_op = tf.summary.merge_all()
        valid_summary_dir = os.path.join(outdir, 'summaries', 'valid')
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)
        
        saver = tf.train.Saver(max_to_keep=num_checkpoint)

        sess.run(tf.global_variables_initializer())

        def run_step(input_data, is_training=True):
            """Run one step of the training process."""
            input_x, input_y, seq_length = input_data
            
            fetches = { 'step': global_step,
                        'cost': classifier.cost,
                        'accuracy': classifier.accuracy,
                        'learning_rate': learning_rate}
            feed_dict = {classifier.input_x: input_x,
                            classifier.input_y: input_y,
                            classifier.seq_length: seq_length}
            # fetches['final_state'] = classifier.final_state
            # fetches['predictions'] = classifier.predictions
            feed_dict[classifier.batch_size] = len(input_x)

            if is_training:
                fetches['train_op'] = train_op
                fetches['summaries'] = train_summary_op
                feed_dict[classifier.keep_prob] = keep_prob
            else:
                fetches['summaries'] = valid_summary_op
                feed_dict[classifier.keep_prob] = 1.0

            vars = sess.run(fetches, feed_dict)
            step = vars['step']
            cost = vars['cost']
            # predictions = vars['predictions']
            accuracy = vars['accuracy']
            summaries = vars['summaries']

            # Write summaries to file
            if is_training:
                train_summary_writer.add_summary(summaries, step)
            else:
                valid_summary_writer.add_summary(summaries, step)
            
            time_str = datetime.datetime.now().isoformat()
            print("{}: step: {}, loss: {:g}, accuracy: {:g}".format(time_str, step, cost, accuracy))
            
            return accuracy
        
        
        print('Start training ...')

        for train_input in train_data:
            run_step(train_input, is_training=True)
            current_step = tf.train.global_step(sess, global_step)
            
            if current_step % evaluate_every_steps == 0:
                print('\nDevlopment Set Validation')
                dev_data = batch_iter(test_data, test_label, test_length, batch_size, 1)
                for dev_input in dev_data:
                    run_step(dev_input, is_training=False)
                print('End Development Set Validation\n')
            
            if current_step % save_every_steps == 0:
                save_path = saver.save(sess, os.path.join(outdir, 'model/clf'), current_step)

        print('\nAll the files have been saved to {}\n'.format(outdir))
