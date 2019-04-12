# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This function loads the pretrained models and runs them through the SentEval library.
# To use, download SentEval from github and place this script in the examples folder.

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import torch
import pickle
import os
sys.path.append('../PyTorch/functions')
from encoders import text_rnn_encoder
from collections import defaultdict

# Set PATHs
PATH_TO_SENTEVAL = '/data/SentEval'
PATH_TO_DATA = '/data/SentEval/data'
glove_loc = '/data/glove.840B.300d.txt'
PATH_TO_ENC = '../flickr_words/results/'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
dict_loc = '../PyTorch/flickr_words/flickr_dict'

# create a dictionary of all the words in the senteval tasks
senteval_dict = defaultdict(int)

def load_obj(loc):
    with open(loc + '.pkl', 'rb') as f:
        return pickle.load(f)

def word_2_index(word_list, batch_size, dict_loc):
    global senteval_dict
    # add words to the Senteval dictionary
    for i, words in enumerate(word_list):
        for j, word in enumerate(words):
            if senteval_dict[word] == 0:
                senteval_dict[word] = len(senteval_dict)
    w_dict = load_obj(dict_loc)
    # filter words that do not occur in the dictionary
    word_list = [[word if word in w_dict else '<oov>' for word in sent] for sent in word_list]
    max_sent_len = max([len(x) for x in word_list])
    text_batch = np.zeros([batch_size, max_sent_len])
    lengths = []
    for i, words in enumerate(word_list):
        lengths.append(len(words))
        for j, word in enumerate(words):
            text_batch[i][j] = w_dict[word]
    return text_batch, lengths

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    # replace empty captions with the out of vocab token
    batch = [sent if sent != [] else ['<oov>'] for sent in batch]
    # add beginning and end of sentence tokens     
    sents = [['<s>'] + x + ['</s>'] for x in batch]
    embeddings = []
    batchsize = len(sents)
    # turn the captions into indices
    sent, lengths = word_2_index(sents, batchsize, dict_loc)
    sort = np.argsort(- np.array(lengths))    
    sent = sent[sort]
    lengths = np.array(lengths)[sort]
    sent = torch.autograd.Variable(torch.cuda.FloatTensor(sent))    
    # embed the captions
    embeddings = params.sent_embedder(sent, lengths)
    embeddings = embeddings.data.cpu().numpy()    
    embeddings = embeddings[np.argsort(sort)]
    return embeddings

dict_len = len(load_obj(dict_loc)) + 3
# create config dictionaries with all the parameters for your encoders
text_config = {'embed':{'num_chars': dict_len, 'embedding_dim': 300, 'sparse': False, 'padding_idx': 0}, 
               'rnn':{'input_size': 300, 'hidden_size': 2048, 'num_layers': 1, 'batch_first': True,
               'bidirectional': True, 'dropout': 0}, 'att':{'in_size': 4096, 'hidden_size': 128, 'heads': 1}}
# create encoder
encoder = text_rnn_encoder(text_config)
for p in encoder.parameters():
    p.requires_grad = False
encoder.cuda()
models = os.listdir(PATH_TO_ENC)
models = [x for x in models if 'caption_model' in x]

for model in models:
    print(model)
    # load pretrained model
    encoder_state = torch.load(os.path.join(PATH_TO_ENC, model))
    encoder.load_state_dict(encoder_state)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.cuda()
    # Set params for SentEval
    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 10}
    params_senteval['sent_embedder'] = encoder
    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    if __name__ == "__main__":
        se = senteval.engine.SE(params_senteval, batcher, prepare)
        transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                          'SICKRelatedness', 'STSBenchmark',
                         ]
        results = se.eval(transfer_tasks)
        print(results)

def save_obj(obj, loc):
    with open(loc + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      ]
    results = se.eval(transfer_tasks)
    print(results)
    save_obj(senteval_dict, 'senteval_dict')

