#! /usr/bin/python
# Author: Jasjeet Dhaliwal

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect, time, reader
import numpy as np
import tensorflow as tf
from reader import *

logging = tf.logging

class TM:
  """LSTM based Topic Model"""
  def __init__(self, config, input_data):
    self.cell_size = config.cell_size
    self.vocab_size = config.vocab_size
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    self.epoch_size = config.epoch_size
    self.num_layers = config.num_layers
    self.scope = config.variable_scope
    self.epochs = config.epochs
    self.cost=0.0
    self.final_state =None
    self.input_data = input_data
    data, labels, weights = self.input_data
  
    def lstm_cell():
      if 'reuse' in inspect.getargspec(
           tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
                 self.cell_size, forget_bias=0.0, state_is_tuple=True,
                 reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
                 self.cell_size, forget_bias=0.0, state_is_tuple=True)
    
    self.net = tf.contrib.rnn.MultiRNNCell(
       [lstm_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    #Get variable to learn word embeddings
    self.embedding = tf.get_variable(
          "embedding", [self.vocab_size, self.cell_size], dtype=tf.float32)
    

    self.initial_state = self.net.zero_state(self.batch_size, tf.float32)
    outputs = []
    state = intial_state
    with tf.variable_scope(self.scope):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(data[:,time_step, :], state)
        outputs.append(cell_output)

    #Reshape outputs for getting logits
    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1,self.cell_size])
    softmax_w = tf.get_variable(
      "softmax_w", [self.cell_size, self.vocab_size], dtype=tf.float32)

    softmax_b = tf.get_variable(
      "softmax_b", [self.vocab_size],dtype=tf.float32)

    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.seq2seq.sequence_loss(
     logits, 
     labels, 
     weights,
     average_across_timesteps=False,
     average_across_batch=True
    )
  
    self.cost = cost = tf.reduce_sum(loss)
    self.final_state = state

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),
                                    config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.frameword.get_or_create_global_step())

    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
    self.lr_update = tf.assign(self.lr, self.new_lr)

    def assign_lr(self, session, lr):
      session.run(self.lr_update, feed_dict={self.new_lr:lr})
 

class HTM:
  """Hierarchial Topic Model with separate word-embeddings at each layer"""

  def __init__(self, word_config, word_data, sen_config, sen_data, doc_config, doc_data)
    self.word_model = TM(word_config, word_data)
    self.sen_model = TM(sen_config, sen_data)
    self.doc_model = TM(doc_config, doc_data)

    
  def run_epoch(self, model, session, eval_op=None):
    "Run one epoch of training on the model"""
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
    }

    if eval_op is not None:
      fetches["eval_op"] = eval_op


    for step in range(model.epoch_size):
      feed_dict = {}
      for i, (c,h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

      vals = session.run(fetches, feed_dict)
      cost = vals["cost"]
      state = vals["final_state"]

      costs+=cost
      iters+=model.num_steps
      
      print ('Costs: {}, Iters: {}'.format(costs, iters))

    return np.exp(costs/iters)

  

  def train_word_model(self, lr, lr_decay):
    sv = tf.train.Supervisor(logidr='./training/word/')
    with sv.managed_session() as session:
      for i in range(self.word_model.epochs):
        self.word_model.assign_lr(session, lr*lr_decay)
        perplexity = self.run_epoch(self.word_model,session,
                         eval_op=self.word_model.train_op)
        print ('Epoch: {}, Perplexity: {}'.format(i, perplexity))
 
  def train_sen_model(self, lr, lr_decay):
    sv = tf.train.Supervisor(logidr='./training/sen/')
    with sv.managed_session() as session:
      for i in range(self.sen_model.epochs):
        self.sen_model.assign_lr(session, lr*lr_decay)
        perplexity = self.run_epoch(self.sen_model,session,
                         eval_op=self.sen_model.train_op)
        print ('Epoch: {}, Perplexity: {}'.format(i, perplexity))
 
            
  def train_word_model(self, lr, lr_decay):
    sv = tf.train.Supervisor(logidr='./training/doc/')
    with sv.managed_session() as session:
      for i in range(self.doc_model.epochs):
        self.doc_model.assign_lr(session, lr*lr_decay)
        perplexity = self.run_epoch(self.doc_model,session,
                         eval_op=self.doc_model.train_op)
        print ('Epoch: {}, Perplexity: {}'.format(i, perplexity))
 

  def train_HTM(init_scale, lr, lr_decay):

    with tf.Graph().as_default():
      initializer = tf.random_uniform_initializer(-init_scale,
                                                    init_scale)
      with tf.name_scope("Train"):
        with tf.variable_scope("HTM", reuse=None, initializer=initializer):
          self.train_word_model(lr, lr_decay)
          self.train_sen_model(lr, lr_decay)
          self.train_doc_model( lr, lr_decay)

  def val_HTM(init_scale, lr, lr_decay):
    with tf.Graph().as_default():
      initializer = tf.random_uniform_initializer(-init_scale,
                                                    init_scale)
      with tf.name_scope("Val"):
        with tf.variable_scope("HTM", reuse=True, initializer=initializer):
          self.val_word_model(lr,lr_decay)
          self.val_sen_model(lr,lr_decay)
          self.val_doc_model(lr, lr_decay)

  def test_HTM(test_data):
    with tf.Graph().as_default():
      with tf.name_scope("Test"):
        with tf.variable_scope("HTM", reuse=True):
          self.test_model(test_data)

class TM:
  """LSTM based Topic Model"""
  def __init__(self, config, input_data):
    self.cell_size = config.cell_size
    self.vocab_size = config.vocab_size
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    self.epoch_size = config.epoch_size
    self.num_layers = config.num_layers
    self.scope = config.variable_scope
    self.epochs = config.epochs
    self.cost=0.0
    self.final_state =None
    self.input_data = input_data
    data, labels, weights = self.input_data
  
    def lstm_cell():
      if 'reuse' in inspect.getargspec(
           tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
                 self.cell_size, forget_bias=0.0, state_is_tuple=True,
                 reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
                 self.cell_size, forget_bias=0.0, state_is_tuple=True)
    
    self.net = tf.contrib.rnn.MultiRNNCell(
       [lstm_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    #Get variable to learn word embeddings
    self.embedding = tf.get_variable(
          "embedding", [self.vocab_size, self.cell_size], dtype=tf.float32)
    

    self.initial_state = self.net.zero_state(self.batch_size, tf.float32)
    outputs = []
    state = intial_state
    with tf.variable_scope(self.scope):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(data[:,time_step, :], state)
        outputs.append(cell_output)

    #Reshape outputs for getting logits
    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1,self.cell_size])
    softmax_w = tf.get_variable(
      "softmax_w", [self.cell_size, self.vocab_size], dtype=tf.float32)

    softmax_b = tf.get_variable(
      "softmax_b", [self.vocab_size],dtype=tf.float32)

    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.seq2seq.sequence_loss(
     logits, 
     labels, 
     weights,
     average_across_timesteps=False,
     average_across_batch=True
    )
  
    self.cost = cost = tf.reduce_sum(loss)
    self.final_state = state

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads,_ = tf.clip_by_global_norm(tf.gradients(cost,tvars),
                                    config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.frameword.get_or_create_global_step())

    self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
    self.lr_update = tf.assign(self.lr, self.new_lr)

    def assign_lr(self, session, lr):
      session.run(self.lr_update, feed_dict={self.new_lr:lr})
 

