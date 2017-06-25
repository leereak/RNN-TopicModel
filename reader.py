#! /usr/bin/python
# Author: Jasjeet Dhaliwal

import numpy as np
import tensorflow as tf

class Reader(object):
  """Read in data to a tensorflow variable"""

  def __init__(self, dfile_path, vocab):
    """ 
    Args: 
      dfilea_path(str): path to file containing data
      vocab(dict): word to id mapping
    """
    with open(dfile_path, 'r') as f:
      self.raw_data = f.read().split('\n')

    del self.raw_data[-1]
    #Split into input data and labels
    data = [datum.split(',')[0] for datum in self.raw_data]
    tmp_labels = [datum.split(',')[1] for datum in self.raw_data]

    #Split each sentence into words
    data = [datum.split() for datum in data]
    #Replace unknown words
    for i, sen in enumerate(data):
      for j, word in enumerate(sen):
        if word.lower() not in vocab:
          data[i][j] = '<unk>'

    #Map words to id
    data = [[vocab[word.lower().strip()] for word in datum] for datum in data]
    
    #Get max sentence length, and pad each sentence with '<PAD>'
    max_len = len(max(data, key=len))
    data = [datum + (max_len-len(datum))*[vocab['<pad>']] for datum in data]

    #Get corresponding labels for each word (we only use the last one)
    labels = [[vocab['<pad>']]*len(datum) for datum in data]
    #Get weights to control loss calculation
    weights = [[0]*len(datum) for datum in data]

    #Set correct topic label
    for idx, label in enumerate(labels):
      label[-1] = vocab[tmp_labels[idx].lower().strip()]
      
    #Set correct loss weights
    for idx, weight in enumerate(weights):
      weight[-1] = 1

    #Store as tf matrices
    self.data = tf.convert_to_tensor(data, dtype=tf.int32)
    self.labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    self.weights = tf.convert_to_tensor(weights, dtype=tf.int16)
