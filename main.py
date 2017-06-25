#!/usr/bin/python
#Author: Jasjeet Dhaliwal

from __future__ import division

from reader import *
from htm import *
from argparse import ArgumentParser
from collections import namedtuple


def get_config(cell_size, vocab_size, text_size, scope, epochs):
  """Get configuration of the HTM"""

  Config = namedtuple(
                "config",
                "cell_size\
                 vocab_size\
                 batch_size\
                 num_steps\
                 epoch_size\
                 num_layers\
                 scope\
                 epochs")

  batch_size = 2
  num_steps = 1
  num_layers = 1

  config = Config(cell_size = cell_size,
                 vocab_size=vocab_size,
                 batch_size = batch_size,
                 num_steps = num_steps,
                 epoch_size = ((text_size // batch_size) -1) // num_steps, 
                 num_layers = num_layers,
                 scope = scope,
                 epochs=epochs)
           
  return config


def run_model(word_file_path, sen_file_path, doc_file_path, test_file_path, vocab_path):
  """Train HTM on data in dfile using vocab mapping
     in vocab_path, and test model on test_file_path
  Args:
    word_file_path(str): path to word level training data
    sen_file_path(str): path to sentence level training data
    doc_file_path(str): path to sentence level training data
    vocab_path(str): path to vocab file
    test_file_path(str): path to test data
  """

  #Build a vocab
  with open(vocab_path, 'r') as v:
    words = v.read().split('\n')
    words = [word.lower().strip() for word in words]
    vocab = dict(zip(words, range(len(words))))

  #Get train and tes data
  word_reader = Reader(word_file_path, vocab)
  sen_reader = Reader(sen_file_path, vocab)
  doc_reader = Reader(doc_file_path, vocab)
  test_reader = Reader(test_file_path, vocab)
  
  #Split data in word, sentence, and testing
  word_data = (word_reader.data, word_reader.labels, word_reader.weights)
  sen_data = (sen_reader.data, sen_reader.labels, sen_reader.weights)
  doc_data = (doc_reader.data, doc_reader.labels, doc_reader.weights)

  cell_size = 10

  word_config = get_config(cell_size, len(vocab), len(word_labels),
                    "word_model", 5)  
  sen_config = get_config(cell_size, len(vocab), len(sen_labels),
                    "sen_model", 5)  
  doc_config = get_config(cell_size, len(vocab), len(doc_labels),
                    "doc_model", 5)  
  
  #Initialize the HTM
  htm = HTM(word_config, word_data, sen_config, sen_data, doc_config, doc_data)
  # Train the HTM 
  htm.train_HTM(0.01, 0.5, 0.004)
  #Test the model

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument("--word_file", help="Provide path to word model training data", default='./word.txt', type=str)
  parser.add_argument("--sen_file", help="Provide path to sen model training data", default='./sen.txt', type=str)
  parser.add_argument("--doc_file", help="Provide path to doc model training data", default='./doc.txt', type=str)
  parser.add_argument("--test_file", help="Provide path test data", default='./test.txt', type=str)
  parser.add_argument("--vocab_file", help="Provide path vocab file", default='./vocab.txt', type=str)

  args = parser.parse_args()
  run_model(args.word_file, args.sen_file, args.test_file, args.vocab_file)
