# -*- coding: utf-8 -*-
"""Copy of Yelp_Experiment_Vocab_Embeddings_Experiment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16SujvXDehWuiOh4z_7QbPNQG7QP2IR3B

# Preprocessing
"""

# from google.colab import files

# uploaded = files.upload()

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import brown
nltk.download('punkt')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from sklearn.model_selection import train_test_split
np.random.seed(0)
torch.manual_seed(0)

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.vocab import GloVe

word_tokenizer = get_tokenizer('basic_english')

class YelpDataset(Dataset):
    """Yelp dataset."""

    def __init__(self, file_name):
        """
        Args:
            file_name: The json file to make the dataset from
        """
        self.df = pd.read_json(file_name, lines=True)

        binary_cat = []
        counter = Counter()
        reviews = []

        #Create target class for each review, build vocab
        for index, row in self.df.iterrows():
            binary_cat.append(row['category'])

            sentences = sent_tokenize(row['text'])
            reviews.append(sentences)
            for i in range(len(sentences)):
              words = word_tokenizer(sentences[i])
              counter.update(words)

        self.vocab = Vocab(counter, min_freq=1)
        self.df['category'] = binary_cat
        self.df['text'] = reviews
        


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        category = self.df.iloc[idx, 0]
        text = self.df.iloc[idx, 1]
        sample = {'category': category, 'text': text}

        return sample

    def get_vocab(self):
      return self.vocab



### DO NOT APPEND ZEROS ###
dataset_train = YelpDataset('dataset_train.json')
dataset_dev = YelpDataset('dataset_dev.json')
dataset_test = YelpDataset('dataset_test.json')

vocab = dataset_train.get_vocab()

def get_indices(sentence, max_sent_len):
  tokens = word_tokenizer(sentence)
  indices = [vocab[token] for token in tokens]
  diff = max_sent_len - len(tokens)
  for i in range(diff):
    indices.append(1)
  return indices


def collate(batch):

  max_num_sents = 0
  max_sent_len = 0
  for sample in batch:
    num_sents = len(sample['text'])
    if num_sents > max_num_sents:
      max_num_sents = num_sents
    for sent in sample['text']:
      if len(word_tokenizer(sent)) > max_sent_len:
        max_sent_len = len(word_tokenizer(sent))
  
  for sample in batch:
    sample['text'] = pad_review(sample['text'], max_num_sents)
    sample['indices']= []
    for sent in sample['text']:
      sample['indices'].append(get_indices(sent, max_sent_len))

  batch_dict = {'text': [], 'indices': [], 'category': []}
  for sample in batch:
    batch_dict['text'].append(sample['text'])
    batch_dict['indices'].append(sample['indices'])
    batch_dict['category'].append(sample['category'])
  batch_dict['indices'] = torch.tensor(batch_dict['indices'])
  batch_dict['category'] = torch.tensor(batch_dict['category'])

  return batch_dict


def pad_review(review, max_len):
  num_sents = len(review)
  for i in range(max_len - num_sents):
    review.append('<pad>')
  return review

batch_size = 8
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, 
                              num_workers=0, collate_fn = collate)
dataloader_dev = DataLoader(dataset_dev, batch_size=batch_size, shuffle=True, 
                              num_workers=0, collate_fn = collate)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, 
                              num_workers=0, collate_fn = collate)


"""# Updated Model"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, weights_matrix):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=1)
        self.embedding.load_state_dict({'weight': weights_matrix})

        self.hidden_size = hidden_size
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs):

        embed_output = self.embedding(inputs)
        embed_output = torch.mean(embed_output, dim=2, keepdim=True).squeeze(2)
        output, hidden = self.gru(embed_output)

        return output, hidden

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.input_size = input_size
        
        self.fcn = nn.Sequential(
            nn.Linear(2*input_size, 10),
            nn.Tanh(),
            nn.Linear(10, 2),
            nn.Tanh()
        )


    def forward(self, x):
        output = self.fcn(x)
        
        return output

## Make weights matrix
vec_size = 300
vocab = dataset_train.get_vocab()
vocab_size = len(vocab)

glove = Word2VecKeyedVectors.load_word2vec_format('glove.w2v.txt')

weights_matrix = np.zeros((vocab_size, vec_size))
i = 0
for word in vocab.itos:
  try:
    weights_matrix[i] = glove[word]
  except KeyError:
    weights_matrix[i] = np.random.normal(scale=0.6, size=(vec_size, ))
  i+=1
  
weights_matrix = torch.tensor(weights_matrix)

from tqdm import tqdm

encoder_output_size = 32
encoder = EncoderRNN(vocab_size, vec_size, encoder_output_size, weights_matrix)
classifier = BinaryClassifier(encoder_output_size)

criterion = nn.CrossEntropyLoss()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)

epochs = 10
total = 0
for n in range(epochs):
    epoch_loss = 0
    count = 0
    for batch in tqdm(dataloader_train):
        encoder.zero_grad()
        classifier.zero_grad()
        loss = 0

        output, hidden = encoder(batch['indices'])

        temp = torch.zeros([batch_size, 2 * encoder_output_size])
        output = output[:,-1,:]
  
        output = classifier(output)
        target = batch['category']

        loss += criterion(output, target)
        epoch_loss+=loss.detach().item()
        loss.backward()

        encoder_optimizer.step()
        classifier_optimizer.step()

  
    if n:
        print("Average loss at epoch {}: {}".format(n, epoch_loss/len(dataloader_train)))

total_correct = 0

for batch in tqdm(dataloader_train):

        output, hidden = encoder(batch['indices'])

        temp = torch.zeros([batch_size, 2 * encoder_output_size])
        output = output[:,-1,:]

        output = classifier(output)

        for i in range(batch_size):
  
          classification = torch.argmax(output[i]).item()
          target = batch['category'][i]
          if target == classification:
             total_correct+=1

print("Accuracy: {}".format(total_correct/(len(dataloader_train) * batch_size)))

total_correct = 0

for batch in tqdm(dataloader_dev):

        output, hidden = encoder(batch['indices'])

        temp = torch.zeros([batch_size, 2 * encoder_output_size])
        output = output[:,-1,:]

        output = classifier(output)

        for i in range(batch_size):
  
          classification = torch.argmax(output[i]).item()
          target = batch['category'][i]
          if target == classification:
             total_correct+=1

print("Accuracy: {}".format(total_correct/(len(dataloader_dev) * batch_size)))

