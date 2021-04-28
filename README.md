# CSR-Call-Scoring

DataLoader_fns.py: contains functions needed to create dataloader object (save vocabulary, pad reviews, collate, get indices of a sentence from the vocab)

DatasetClasses.py: contains the classes that define the Yelp dataset and the Call Transcripts dataset

Inference_fns.py: contains functions that make inferences (getting accuracy, predicting for a new data point)

MainCalls.py: creates datasets from call transcripts for train, test, and dev, saves vocabulary, creates dataloaders, creates initial word embeddings matrix, trains model, runs model on test set and reports accuracies, saves the final model after training

MainYelp.py: creates datasets from Yelp Dataset for train, test, and dev, saves vocabulary, creates dataloaders, creates initial word embeddings matrix, trains model, runs model on test set and reports accuracies, saves the final model after training

MainCallsInference.py: loads in saved model and uses it to predict whether a new call is good or bad (cmd line argument is the file name that you want to run inference on)

Models.py: contains class definitions for the encoder and binary classifier pytorch models

Preprocessing.py: contains functions for call transcript preprocessing (takes in call transcript and returns an array of cleaned tokenized sentences)

TrainModel.py: contains function to train the model and reports accuracy and loss at each epoch
