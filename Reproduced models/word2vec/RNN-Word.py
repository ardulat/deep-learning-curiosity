# coding: utf-8


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
import time
import sys
import pickle
import argparse
import copy


class Config(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.90
  learning_rate = 0.7
  init_scale = 0.1
  num_epochs = 70
  max_epoch = 3
  word_vocab_size = 0 # to be determined later

  # RNN hyperparameters
  num_steps = 35
  hidden_size = 50


def read_data(config):
  '''read data sets, construct all needed structures and update the config'''
  word_data = open('data/ptb/train.txt', 'r').read().replace('\n', '<eos>').split()
  words = list(set(word_data))

  word_data_size, word_vocab_size = len(word_data), len(words)
  print('data has %d words, %d unique' % (word_data_size, word_vocab_size))
  config.word_vocab_size = word_vocab_size

  word_to_ix = { word:i for i,word in enumerate(words) }
  ix_to_word = { i:word for i,word in enumerate(words) }

  def get_word_raw_data(input_file):
    data = open(input_file, 'r').read().replace('\n', '<eos>').split()
    return [word_to_ix[w] for w in data]

  train_raw_data = get_word_raw_data('data/ptb/train.txt')
  valid_raw_data = get_word_raw_data('data/ptb/valid.txt')
  test_raw_data = get_word_raw_data('data/ptb/test.txt')

  return train_raw_data, valid_raw_data, test_raw_data


class batch_producer(object):
  '''Slice the raw data into batches'''
  def __init__(self, raw_data, batch_size, num_steps):
    self.raw_data = raw_data
    self.batch_size = batch_size
    self.num_steps = num_steps
    
    self.batch_len = len(self.raw_data) // self.batch_size
    self.data = np.reshape(self.raw_data[0 : self.batch_size * self.batch_len],
                           (self.batch_size, self.batch_len))
    
    self.epoch_size = (self.batch_len - 1) // self.num_steps
    self.i = 0
  
  def __next__(self):
    if self.i < self.epoch_size:
      # batch_x and batch_y are of shape [batch_size, num_steps]
      batch_x = self.data[::, 
          self.i * self.num_steps : (self.i + 1) * self.num_steps : ]
      batch_y = self.data[::, 
          self.i * self.num_steps + 1 : (self.i + 1) * self.num_steps + 1 : ]
      self.i += 1
      return (batch_x, batch_y)
    else:
      raise StopIteration()

  def __iter__(self):
    return self


config = Config()
train_raw_data, valid_raw_data, test_raw_data = read_data(config)


# get hyperparameters
batch_size = config.batch_size
num_steps = config.num_steps
init_scale = config.init_scale
word_emb_dim = hidden_size = config.hidden_size
word_vocab_size = config.word_vocab_size

initializer = tf.random_uniform_initializer(-config.init_scale,
                                            config.init_scale)

# language model 
with tf.variable_scope('model', initializer=initializer):
    # embedding matrix
    word_embedding = tf.get_variable("word_embedding", [word_vocab_size, word_emb_dim])

    # placeholders for training data and labels
    x = tf.placeholder(tf.int32, [batch_size, num_steps])
    y = tf.placeholder(tf.int32, [batch_size, num_steps])

    # we first embed words ...
    words_embedded = tf.nn.embedding_lookup(word_embedding, x)

    # ... and then process it with a stack of two LSTMs
    rnn_input = tf.unstack(words_embedded, axis=1)

    # basic RNN cell
    cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    outputs, state = tf.contrib.rnn.static_rnn(
        cell, 
        rnn_input, 
        dtype=tf.float32, 
        initial_state=init_state)
    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])

    # softmax layer    
    weights = tf.get_variable('weights', [hidden_size, word_vocab_size], dtype=tf.float32)
    biases = tf.get_variable('biases', [word_vocab_size], dtype=tf.float32)

    logits = tf.matmul(output, weights) + biases
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(y, [-1])],
        [tf.ones([batch_size * num_steps], dtype=tf.float32)])
    cost = tf.reduce_sum(loss) / batch_size

# training
lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.apply_gradients(zip(grads, tvars), 
                                     global_step = tf.contrib.framework.get_or_create_global_step())

new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
lr_update = tf.assign(lr, new_lr)

#session.run(lr_update, feed_dict={new_lr: lr_value})

def model_size():
  '''finds the total number of trainable variables a.k.a. model size'''
  params = tf.trainable_variables()
  size = 0
  for x in params:
    sz = 1
    for dim in x.get_shape():
      sz *= dim.value
    size += sz
  return size


def run_epoch(sess, raw_data, config, is_train=False, lr=None):
  start_time = time.time()
  if is_train: sess.run(lr_update, feed_dict={new_lr: lr})

  iters = 0
  costs = 0
  state_val = sess.run(init_state)

  batches = batch_producer(raw_data, config.batch_size, config.num_steps)

  for (batch_x, batch_y) in batches:
    # run the model on current batch
    if is_train:
      _, c, state_val = sess.run(
          [train_op, cost, state],
          feed_dict={x: batch_x, y: batch_y, 
                     init_state: state_val})
    else:
      c, state_val = sess.run([cost, state],
          feed_dict={x: batch_x, y: batch_y, 
                     init_state: state_val})

    costs += c
    step = iters // config.num_steps
    if is_train and step % (batches.epoch_size // 10) == 10:
      print('%.3f' % (step * 1.0 / batches.epoch_size), end=' ')
      print('train ppl = %.3f' % np.exp(costs / iters), end=', ')
      print('speed =', 
          round(iters * config.batch_size / (time.time() - start_time)), 
          'wps')
    iters += config.num_steps
  
  return np.exp(costs / iters)


if __name__ == '__main__':
 
  print('Model size is: ', model_size())

  saver = tf.train.Saver()
    
  num_epochs = config.num_epochs
  init = tf.global_variables_initializer()
  learning_rate = config.learning_rate

  with tf.Session() as sess:
    sess.run(init)
    prev_valid_ppl = float('inf')
    best_valid_ppl = float('inf')

    for epoch in range(num_epochs):
      train_ppl = run_epoch(
          sess, train_raw_data, config, is_train=True, 
          lr=learning_rate)
      print('epoch', epoch + 1, end = ': ')
      print('train ppl = %.3f' % train_ppl, end=', ')
      print('lr = %.3f' % learning_rate, end=', ')

      # Get validation set perplexity
      valid_ppl = run_epoch(
          sess, valid_raw_data, config, is_train=False)
      print('valid ppl = %.3f' % valid_ppl)
        
      # Update the learning rate if necessary
      if epoch + 2 > config.max_epoch: learning_rate *= config.lr_decay
        
      # Save model if it gives better valid ppl
      if valid_ppl < best_valid_ppl:
        save_path = saver.save(sess, 'saves/model.ckpt')
        print('Valid ppl improved. Model saved in file: %s' % save_path)
        best_valid_ppl = valid_ppl

  '''Evaluation of a trained model on test set'''
  with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, 'saves/model.ckpt')
    print('Model restored.')

    # Get test set perplexity
    test_ppl = run_epoch(sess, test_raw_data, test_config, is_train=False)
    print('Test set perplexity = %.3f' % test_ppl)

