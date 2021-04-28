# -*- coding: utf-8 -*-
"""MainCallsInference

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lNa45_LJoH8SpVb3A_-VcTo8k--sHOV4
"""

import torch
import pickle
from Inference_fns import predict
import sys

if len(sys.argv) != 2:
  print("Improper number of arguments")
  print("Only command line argument should be file name")
else:
  encoder = torch.load('encoder_calls.model')
  classifier = torch.load('classifier_calls.model')

  file = open("vocab",'rb')
  vocab = pickle.load(file)
  file.close()

  prediction = predict(sys.argv[1], encoder, classifier, vocab)
  print(prediction)
  if prediction == 1:
    print("Call predicted as good")
  else:
    print("Call predicted as bad")