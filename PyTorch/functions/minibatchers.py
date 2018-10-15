#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:06:06 2018

@author: danny
minibatchers for the neural network training. There are three batchers for speech, tokens and raw text.
Each batcher has a 5 fold version as many image captioning databases have multiple (5) captions per image.
The batchers also return the lenghts of the captions in the batch so it can be used with torch
pack_padded_sequence.
"""
import numpy as np
from prep_text import char_2_index, word_2_index

# visual and text should be the names of the feature nodes in the h5 file, chars is the maximum sentence length in characters.
# default is 260, to accomodate the max lenght found in mscoco. The max lenght in flickr is 200 
def iterate_char(f_nodes, batchsize, visual, text, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        caption = []
        images = []
        for ex in excerpt:
            # extract and append the vgg16 features
            images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
            # extract the audio features
            cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
            cap = cap.decode('utf-8')
            # append an otherwise unused character as a start of sentence character and 
            # convert the sentence to lower case.
            caption.append(cap)
        # converts the sentence to character ids. 
        caption, lengths = char_2_index(caption, batchsize)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, caption, lengths

# slightly different from the raw text loader, also takes a dictionary location. Max words default is 60 to accomodate mscoco.
def iterate_tokens(f_nodes, batchsize, visual, text, dict_loc, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
        # take a batch of nodes of the given size               
        excerpt = f_nodes[start_idx:start_idx + batchsize]
        caption = []
        images = []
        for ex in excerpt:
            # extract and append the vgg16 features
            images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
            # extract the audio features
            cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
            cap = [x.decode('utf-8') for x in cap]
            # append an otherwise unused character as a start of sentence character and 
            # convert the sentence to lower case.
            caption.append(cap)
        # converts the sentence to character ids. 
        caption, lengths = word_2_index(caption, batchsize, dict_loc)
        images_shape = np.shape(images)
        # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
        images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
        yield images, caption, lengths

# iterate over text input. the value for chars indicates the max sentence lenght in characters. Keeps track 
# of the unpadded senctence lengths to use with pytorch's pack_padded_sequence.
def iterate_char_5fold(f_nodes, batchsize, visual, text, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0,5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            caption = []
            images = []
            for ex in excerpt:
                # extract and append the vgg16 features
                images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                # extract the audio features
                cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
                cap = cap.decode('utf-8')
                # append an otherwise unused character as a start of sentence character and 
                # convert the sentence to lower case.
                caption.append(cap)
            # converts the sentence to character ids. 
            caption, lengths = char_2_index(caption, batchsize)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, caption, lengths

# iterate over text input. the value for chars indicates the max sentence lenght in characters. Keeps track 
# of the unpadded senctence lengths to use with pytorch's pack_padded_sequence. slightly different from the raw text loader
# as we need a word_2_index function and this also takes a dictionary
def iterate_tokens_5fold(f_nodes, batchsize, visual, text, dict_loc, shuffle=True):
    if shuffle:
        # optionally shuffle the input
        np.random.shuffle(f_nodes)
    for i in range(0,5):
        for start_idx in range(0, len(f_nodes) - batchsize + 1, batchsize):
            # take a batch of nodes of the given size               
            excerpt = f_nodes[start_idx:start_idx + batchsize]
            caption = []
            images = []
            for ex in excerpt:
                # extract and append the vgg16 features
                images.append(eval('ex.' + visual + '._f_list_nodes()[0].read()'))
                # extract the audio features
                cap = eval('ex.' + text + '._f_list_nodes()[i].read()')
                # add begin of sentence and end of sentence tokens
                cap = ['<s>'] + [x.decode('utf-8') for x in cap] + ['</s>']
                                
                caption.append(cap)
            # converts the sentence to character ids. 
            caption, lengths = word_2_index(caption, batchsize, dict_loc)
            images_shape = np.shape(images)
            # images should be shape (batch_size, 1024). images_shape[1] is collapsed as the original features are of shape (1,1024) 
            images = np.float64(np.reshape(images,(images_shape[0],images_shape[2])))
            yield images, caption, lengths

