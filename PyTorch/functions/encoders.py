#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:54:07 2018

@author: danny
"""

from costum_layers import RHN, attention, multi_attention
from load_embeddings import load_word_embeddings
import torch
import torch.nn as nn

# gru encoder for characters and tokens
class text_rnn_encoder(nn.Module):
    def __init__(self, config):
        super(text_gru_encoder, self).__init__()
        embed = config['embed']
        rnn= config['rnn']
        att = config ['att'] 
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
        self.RNN = nn.GRU(input_size = rnn['input_size'], hidden_size = rnn['hidden_size'], 
                          num_layers = rnn['num_layers'], batch_first = rnn['batch_first'],
                          bidirectional = rnn['bidirectional'], dropout = rnn['dropout'])
        self.att = multi_attention(in_size = att['in_size'], hidden_size = att['hidden_size'], n_heads = att['heads'])
        
    def forward(self, input, l):
        # embedding layers expect Long tensors
        x = self.embed(input.long())
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = torch.nn.utils.rnn.pack_padded_sequence(x, l, batch_first=True)
        x, hx = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of words occuring in the training data
        # and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data)         

# the network for embedding the visual features
class img_encoder(nn.Module):
    def __init__(self, config):
        super(img_encoder, self).__init__()
        linear = config['linear']
        self.norm = config['norm']
        self.linear_transform = nn.Linear(in_features = linear['in_size'], out_features = linear['out_size'])
        nn.init.xavier_uniform(self.linear_transform.weight.data)
    def forward(self, input):
        x = self.linear_transform(input)
        if self.norm:
            return nn.functional.normalize(x, p=2, dim=1)
        else:
            return x

######################################################################################################
# network concepts and experiments and networks by others
######################################################################################################

# simple encoder that just sums the word embeddings of the tokens
class bow_encoder(nn.Module):
    def __init__(self, config):
        super(bow_encoder, self).__init__()
        embed = config['embed']
        self.embed = nn.Embedding(num_embeddings = embed['num_chars'], 
                                  embedding_dim = embed['embedding_dim'], sparse = embed['sparse'],
                                  padding_idx = embed['padding_idx'])
    def forward(self, input, l):
        # embedding layers expect Long tensors
        x = self.embed(input.long())
        return x.sum(2)
    
    def load_embeddings(self, dict_loc, embedding_loc):
        # optionally load pretrained word embeddings. takes the dictionary of words occuring in the training data
        # and the location of the embeddings.
        load_word_embeddings(dict_loc, embedding_loc, self.embed.weight.data) 

# the convolutional character encoder described by Wehrmann et al. 
class conv_encoder(nn.Module):
    def __init__(self):
        super(conv_encoder, self).__init__()
        self.Conv1d_1 = nn.Conv1d(in_channels = 20, out_channels = 512, kernel_size = 7,
                                 stride = 1, padding = 3, groups = 1)
        self.Conv1d_2 = nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 5,
                                 stride = 1, padding = 2, groups = 1)
        self.Conv1d_3 = nn.Conv1d(in_channels = 512, out_channels = 512, kernel_size = 3,
                                 stride = 1, padding = 1, groups = 1)
        self.relu = nn.ReLU()
        self.embed = nn.Embedding(num_embeddings = 100, embedding_dim = 20,
                                  sparse = False, padding_idx = 0)
        self.Pool = nn.AdaptiveMaxPool1d(output_size = 1, return_indices=False)
        self.linear = nn.Linear(in_features = 512, out_features = 512)
    def forward(self, input, l):
        x = self.embed(input.long()).permute(0,2,1)
        x = self.relu(self.Conv1d_1(x))
        x = self.relu(self.Conv1d_2(x))
        x = self.relu(self.Conv1d_3(x))
        x = self.linear(self.Pool(x).squeeze())
        return nn.functional.normalize(x, p = 2, dim = 1)
