# -*- coding: utf-8 -*-
"""Preprocessing

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18SR3aBGRcfjDrJNWhEIbIPsUI_d6czXv
"""

import nltk
from nltk.tokenize import word_tokenize
import unidecode
import contractions

#Takes a call transcript as argument and returns a cleaned version of the text
def preprocess_transcript(filename):
    try:
        file_in = open(filename, 'r')
        try:
            text = file_in.read()
            file_in.close()
            text = text.splitlines()
            for utterance in text:
                utterance = clean_text(utterance)
            return text
        except e:
            print("Failed to read transcript")
    except e:
        print("Failed to open transcript")


#Cleans a chunk of text and prepares it for analysis
def clean_text(text):

    #Get rid of accents
    text = unidecode.unidecode(text)

    #Expand contractions
    text = contractions.fix(text)

    #Remove special characters
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)

    #Make all whitespace into space
    text = ' '.join(text.split())

    #Make everything lowercase
    text = text.lower()
    
    return text