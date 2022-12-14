"""
Helper functions for stimuli generation.
"""
import numpy as np
import random
import string 


# Probability of each letter in english text
letterFrequency = {'E' : 12.0,'T' : 9.10,'A' : 8.12,'O' : 7.68,'I' : 7.31,'N' : 6.95,'S' : 6.28,'R' : 6.02,'H' : 5.92,'D' : 4.32,'L' : 3.98,'U' : 2.88,'C' : 2.71,'M' : 2.61,'F' : 2.30,'Y' : 2.11,'W' : 2.09,'G' : 2.03,'P' : 1.82,'B' : 1.49,'V' : 1.11,'K' : 0.69,'X' : 0.17,'Q' : 0.11,'J' : 0.10,'Z' : 0.07 }

def not_space(text):
    """
    Searches text for random letter index, which is not a space character.
    """
    r = random.randrange(0, len(text))
    while text[r] == ' ':
        r = random.randrange(0, len(text))
    return r

def zero_order(l):
    """
    Generates zeroth order words (random letter sequence) with given 'l' length.
    """
    word0 = ''
    r = np.random.randint(97, 123, l)
    for i in r:
        word0 = word0 + chr(i)
    #print(word0)
    return word0

def first_order(text, l):
    """
    Generates first order words (with the help of letterFrequency) with given 'l' length.
    """
    word1 = ''
    for i in range(l):
        r = not_space(text)
        word1 = word1 + text[r]
    #print(word1)
    return word1

def second_order(text, l):
    """
    Generates second order words (with the frequency of letter doublets in english text) with given 'l' length.
    """
    r = not_space(text) 
    letter = text[r]
    word2 = letter
    for i in range(l-1):
        start = random.randrange(0, len(text)) 
        text1 = text[0:start]
        text2 = text[start:len(text)]
        text_new = text2 + text1
        ind = text_new.find(letter)

        if ind == -1:   
            second_order(text, l)
            return ""

        while text_new[ind + 1] == ' ':    
            text_new = text_new[ind+1:len(text2)]
            ind = text_new.find(letter)

        letter = text_new[ind + 1]
        word2 += letter
    #print(word2)
    return word2

def third_order(text, l): 
    """
    Generates third order words (with the frequency of letter triplets in english text) with given 'l' length.
    """
    while True:
        r = not_space(text)  
        doublet = text[r:r+2]
        if doublet.find(' ') == -1 and text[r-1] == ' ':
            break

    word3 = doublet
    for i in range(l-2):
        start = random.randrange(0, len(text)) 
        text1 = text[0:start]
        text2 = text[start:len(text)]
        text_new = text2 + text1
        ind = text_new.find(doublet) 

        if ind == -1:   
            third_order(text, l)
            return ""

        while text_new[ind + 2] == ' ':    
            text_new = text_new[ind+2:len(text_new)]
            ind = text_new.find(doublet)

        letter = text_new[ind + 2]
        word3 += letter
        doublet = word3[-2:]

    #print(word3)
    return word3


def fourth_order(text, l):
    """
    Generates fourth order words (with the frequency of letter quadruplet in english text) with given 'l' length.
    """
    while True:
        r = not_space(text) 
        triplet = text[r:r+3]
        if triplet.find(' ') == -1:
            break

    word4 = triplet

    if l == 1:
        word4 = word4[0]
    elif l == 2:
        word4 = word4[0:2]
    else:
        for i in range(l-3):
            start = random.randrange(0, len(text))
            text1 = text[0:start]
            text2 = text[start:len(text)]
            text_new = text2+text1
            ind = text_new.find(triplet)

            if ind == -1 or ind >= len(text_new)-5:   
                fourth_order(text,l)
                return ""

            while text_new[ind + 3] == ' ':    
                text_new = text_new[ind+3:len(text_new)]
                ind = text_new.find(triplet)
                if ind == -1 or ind >= len(text_new) - 5:
                    fourth_order(text,l)
                    return ""

            letter = text_new[ind + 3]
            word4 += letter
            triplet = word4[-3:]

        #print(word4)
    return word4

def word_order(text, l):
    """
    Gives back random words with given 'l' length.
    """
    words = text.rsplit()
    r = random.randrange(0, len(words))
    while len(words[r]) != l:
        r = random.randrange(0, len(words))

    return words[r]


def difference_and_new_word(w1, w2, mode="totallyrandom"): # modes: totallyrandom, firstorder, VAE_chars, notVAEplace_1st
    """
    Searches the difference between original text and VAE reconstruction.
    """
    diff = 0
    rand = ""
    VAE_c = []
    for h, s in enumerate(zip(w1,w2)):
        if s[0] != s[1]:
            diff +=1
            if mode == "firstorder":
                c = random.choices(list(letterFrequency.keys()), weights = list(letterFrequency.values()))[0].lower()
                while c == s[0] or c == s[1]:
                    c = random.choices(list(letterFrequency.keys()), weights = list(letterFrequency.values()))[0].lower()
                s = c
            elif mode == "VAE_chars" or mode == "notVAEplace_1st":
                VAE_c.append((h,s[1]))
            else:
                c = random.randrange(97, 123)
                while chr(c) == s[0] or chr(c) == s[1]:
                    c = random.randrange(97, 123)
                s = chr(c)
        rand += s[0]
    
    if mode == "totallyrandom": 
        idx = random.sample(range(7), diff)
        ws_new = list(w1)
        for d in range(diff):
            char = random.choices(string.ascii_lowercase)
            while char == ws_new[idx[d]]:
                char = random.choices(string.ascii_lowercase)
            ws_new[idx[d]] = char[0]
        rand = "".join(ws_new)

    if mode == "VAE_chars":
        if diff < len(w1)/2:
            w1 = list(w1)
            for (h,c) in VAE_c:
                new_h = random.randrange(1,7)
                while new_h in list(np.asarray(VAE_c).T[0].astype(int)) or w1[new_h] == c:
                    new_h = random.randrange(1,7)
                w1[new_h] = c
            rand = "".join(w1)
        else:
            print("This is not working on too many orig-vae difference!")
            exit()

    if mode == "notVAEplace_1st":
        if diff < len(w1)/2:
            VAE_hs = list(np.asarray(VAE_c).T[0].astype(int))
            w1 = list(w1)
            for h in VAE_c:
                new_h = random.randrange(1,6)
                c = random.choices(list(letterFrequency.keys()), weights = list(letterFrequency.values()))[0].lower()
                while new_h in VAE_hs or w1[new_h] == c:
                    new_h = random.randrange(1,6)
                    c = random.choices(list(letterFrequency.keys()), weights = list(letterFrequency.values()))[0].lower()
                VAE_hs.append(new_h)
                w1[new_h] = c
            rand = "".join(w1)
        else:
            print("This is not working on too many orig-vae difference!")
            exit()

    return rand, diff
