#!/usr/bin/env python
# coding: utf-8
"""
Functions for generating stimuli for the experiment.
Experiment consists of so-called 'original' words, which are lower order english-like letter sequences; 'VAE' reconstructions of 'orig' words;
and 'filler' words, which are rather random distortions of origs. 
"""

import numpy as np
import string
import os
from stim_generation_helper import *
from traindata_generation import *

def orders_generation(book, output_dir, num=50, ords=[3], length=7):
    """
    Saves 'num' numbers of generated words with 'l' length in given orders [ords] into txt files.
    """
    realwords = loading(book)

    if 0 in ords:
        file = open(os.path.join(output_dir, "w"+ str(length) +"_0.txt"), "w+", encoding = 'utf-8')
        i = 0
        while i < num:
            word = zero_order(length)
            if word != "":
                file.write(word + '\n')
                i +=1
    if 1 in ords:
        file = open(os.path.join(output_dir, "w"+ str(length) +"_1.txt"), "w+", encoding = 'utf-8')
        i = 0
        while i < num:
            word = first_order(realwords, length)
            if word != "":
                file.write(word + '\n')
                i +=1  
    if 2 in ords:
        file = open(os.path.join(output_dir, "w"+ str(length) +"_2.txt"), "w+", encoding = 'utf-8')
        i = 0
        while i < num:
            word = second_order(realwords, length)
            if word != "":
                file.write(word + '\n')
                i +=1
    if 3 in ords:
        file = open(os.path.join(output_dir, "w"+ str(length) +"_3.txt"), "w+", encoding = 'utf-8')
        i = 0
        while i < num:
            word = third_order(realwords, length)
            if word != "":
                file.write(word + '\n')
                i +=1
    if 4 in ords:
        file = open(os.path.join(output_dir, "w"+ str(length) +"_4.txt"), "w+", encoding = 'utf-8')
        i = 0
        while i < num:
            word = fourth_order(realwords, length)
            if word != "":
                file.write(word + '\n')
                i +=1
    if 5 in ords:
        file = open(os.path.join(output_dir, "w"+ str(length) +"_word.txt"), "w+", encoding = 'utf-8')
        i = 0
        while i < num:
            word = word_order(realwords, length)
            if word != "":
                file.write(word + '\n')
                i +=1

def HD_sorting(origFile, vaeFile, HD=2):
    """
    Choses orig and VAE words with a given 'HD' Hamming-distance.
    """
    with open(origFile, "r", encoding = 'utf-8', errors='ignore') as ofile, open(vaeFile, "r", encoding = 'utf-8', errors='ignore') as vfile:
        words_orig = ofile.read().split()
        words_VAE = vfile.read().split()
    
    with open(origFile[:-4]+"_HD"+str(HD)+".txt", "w+", encoding = 'utf-8', errors='ignore') as sort_ofile, open(vaeFile[:-4]+"_HD"+str(HD)+".txt", "w+", encoding = 'utf-8', errors='ignore') as sort_vfile:     
        for ws in zip(words_orig, words_VAE): 
            _, dist = difference_and_new_word(ws[0], ws[1])
            if dist == HD:
                sort_ofile.write(ws[0] + '\n')
                sort_vfile.write(ws[1] + '\n')


def filler_making(origFile, vaeFile, fillerFile, mode="totallyrandom" ): #modes: totallyrandom, firstorder, VAE_chars, notVAEplace_1st
    """
    Generates 'fillers' with the same number of difference as in the VAE distortions and writes them in separate files.
    Modes:
    'totallyrandom' - randomly choses places and replaces letter with new one
    'first order' - errs at the same place as the VAE, and replaces letter in line with the letter frequency in natural language
    'VAE_chars' - uses the same letters as the VAE did for replacement of original letter, but at different places
    'notVAEplace_1st' - uses 1st order letters, but at different places as the VAE erred
    """
    with open(origFile, "r", encoding = 'utf-8', errors='ignore') as ofile, open(vaeFile, "r", encoding = 'utf-8', errors='ignore') as vfile:
        words_orig = ofile.read().split()
        words_VAE = vfile.read().split()
        with open(fillerFile, "w+", encoding = 'utf-8', errors='ignore') as f:
            for ws in zip(words_orig, words_VAE):
                new_word, _ = difference_and_new_word(ws[0], ws[1], mode)
                print(new_word)
                f.write(new_word + '\n')

def cleaning(origFile, vaeFile, fillerFile):
    """
    Cleans already generated orig, VAE, and filler files of first and last letter distortions, real words, and duplicates.
    Writes them in new separate files.
    """
    with open(origFile, "r", encoding = 'utf-8', errors='ignore') as ofile, open(vaeFile, "r", encoding = 'utf-8', errors='ignore') as vfile, open(fillerFile, "r", encoding = 'utf-8', errors='ignore') as ffile:
        words_orig = ofile.read().split()
        words_VAE = vfile.read().split()
        words_filler = ffile.read().split()

    realwords = loading("books_for_model_training/15books_eng_for_test.txt").rsplit() 
    V = []
    o = []
    f = []
    with open(origFile[:-4] + "_clean.txt", "w", encoding = 'utf-8', errors='ignore') as ofile, open(vaeFile[:-4] + "_clean.txt", "w", encoding = 'utf-8', errors='ignore') as vfile, open(fillerFile[:-4] + "_clean.txt", "w", encoding = 'utf-8', errors='ignore') as ffile:    
        for ws in zip(words_orig, words_VAE, words_filler):
            if ws[0][0] == ws[1][0] and ws[0][-1] == ws[1][-1] and ws[0][0] == ws[2][0] and ws[0][-1] == ws[2][-1] and ws[0] not in realwords and ws[1] not in realwords and ws[2] not in realwords and ws[0] not in o and ws[1] not in V and ws[2] not in f and ws[1] not in o and ws[2] not in V and ws[0] not in f and ws[2] not in o and ws[0] not in V and ws[1] not in f:
                o.append(ws[0])
                V.append(ws[1])
                f.append(ws[2])
                ofile.write(ws[0] + '\n')
                vfile.write(ws[1] + '\n')
                ffile.write(ws[2] + '\n')
            else:
                print(ws)


