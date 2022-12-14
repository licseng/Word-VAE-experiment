#!/usr/bin/env python
# coding: utf-8
"""
Helper functions to make traindata, converting input to one-hot, and converting back to readable words.
"""

import numpy as np
import string
import random
import unicodedata
from sklearn.model_selection import train_test_split

LEN_ABC = 27 # Number of letters

def remove_accents(input_str):
    """
    Removes accents from text.
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def loading(file):
    """
    Loads a txt book as 'space' separated words, cleans it of accents, weird characters, sets letters to lowcase.
    """
    file = open(file, "r", encoding = 'utf-8', errors='ignore') 
    data = file.read()
    file.close()
    proc_data = ''
    for x in data:
        x = x.lower()
        if x.isalpha() or x == ' ':
            proc_data = proc_data + x
        elif x == '\n':
            proc_data = proc_data + ' '
    return remove_accents(proc_data)

def text_to_onehot(words, l):
    """
    Changes words to one-hot encoding, if word is shorter than given 'l', padding is added (space) randomly distributed around the word.
    """
    one_hot_matrices = np.zeros((len(words), l, LEN_ABC), dtype=np.int8)
    
    for i, w in enumerate(words):
        w = list(w)
        w = [ord(x) - 97 for x in w if ord(x) - 97 < LEN_ABC]  
        if len(w) >= l:
            wo = np.asarray(w[0:l]) 
        else:
            r = np.random.randint(0, l - len(w)+1)
            wo = np.pad(w, (r, l - len(w)-r), 'constant', constant_values=26)
        wo = wo.astype(np.int64)

        one_hot_matrices[i] = np.eye(LEN_ABC)[wo]
        
    return one_hot_matrices

def onehot_to_word(onehot):
    """
    Converts back beta-VAE output categorical distribution from one-hot encoding to letters.
    """
    numbers = np.argmax(onehot, axis=1)
    abc = list(string.ascii_lowercase)
    abc.append("_")
    word_list = [abc[i] for i in numbers]
    word = ""
    for ch in word_list: 
        word += ch  
    return word

def training_data_making(outFile_name, book_list, randomwords_num=0, word_size=7):
    """
    Gets a list full of books in txt form, and saves it in one-hot encoded form to a .npy file. 
    Does train-test split, and saves in separate files.
    """
    landmore = []
    for book in book_list: 
        data = loading(book)
        data = data.split()
        for w in data:
            if len(w) == word_size:
                landmore.append(w)
            elif len(w) > word_size:
                landmore.append(w[:word_size])
        print(len(landmore))
    train_txt, test_txt = train_test_split(landmore, test_size=0.2)
    
    for i in range(randomwords_num):
        randomword = ''.join(random.choice(string.ascii_lowercase) for x in range(word_size))
        landmore.append(randomword)
        print(len(landmore))
        random.shuffle(landmore)
        
    np.save(outFile_name+ "_train_"+str(word_size)+"long.npy", text_to_onehot(train_txt, word_size))
    np.save(outFile_name+ "_test_"+str(word_size)+"long.npy", text_to_onehot(test_txt, word_size))
