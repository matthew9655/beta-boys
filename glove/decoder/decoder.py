# Imports
%matplotlib inline
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from torch.autograd import Variable

torch.manual_seed(1)

from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os

# TODO: if you are running the code on GPU, you can remove CPU section
model = None
losses = None
word_to_ix = None
recon = None


class Glove(nn.Module):

    def __init__(self, vocab_size, comat, embedding_size, x_max, alpha):
        super(Glove, self).__init__()
        
        # embedding matrices
        self.embedding_V = nn.Embedding(vocab_size, embedding_size) # embedding matrix of center words
        self.embedding_U = nn.Embedding(vocab_size, embedding_size) # embedding matrix of context words

        # biases
        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)
        
        # initialize all params
        for params in self.parameters():
            nn.init.uniform_(params, a = -0.5, b = 0.5)
            
        #hyperparams
        self.x_max = x_max
        self.alpha = alpha
        self.comat = comat
    
    def forward(self, center_word_lookup, context_word_lookup):
        # indexing into the embedding matrices
        center_embed = self.embedding_V(center_word_lookup)
        target_embed = self.embedding_U(context_word_lookup)

        center_bias = self.v_bias(center_word_lookup).squeeze(1)
        target_bias = self.u_bias(context_word_lookup).squeeze(1)

        # elements of the co-occurence matrix
        co_occurrences = torch.tensor([self.comat[center_word_lookup[i].item(), context_word_lookup[i].item()]
                                       for i in range(BATCH_SIZE)])
        
        # weight_fn applied to non-zero co-occurrences
        weights = torch.tensor([self.weight_fn(var) for var in co_occurrences])

        # the loss as described in the paper
        loss = torch.sum(torch.pow((torch.sum(center_embed * target_embed, dim=1)
            + center_bias + target_bias) - torch.log(co_occurrences), 2) * weights)
        
        return loss
        
    def weight_fn(self, x):
        # the proposed weighting fn
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1
        
    def embeddings(self):
        # "we choose to use the sum W + W_tilde as our word vectors"
        return self.embedding_V.weight.data + self.embedding_U.weight.data

# Helper functions

def get_word(word, model, word_to_ix):
    """
    returns the embedding that belongs to the given word (str)
    """
    return model.embeddings()[word_to_ix[word]]

def closest(vec, word_to_ix, n=10):
    """
    finds the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w, model, word_to_ix))) for w in word_to_ix]
    return sorted(all_dists, key=lambda t: t[1])[:n]

# Decoder helper functions
def word_vectors(model, word_to_ix):
    words = []
    embedding = []
    for w in word_to_ix:
        words.append(words)
        embedding.append(get_word(w, model, word_to_ix))
    return words, embedding

def min_dist(vec, word_vec):
    idx = 0
    min = float('inf')
    for i in range(len(word_vec)):
        dist = torch.dist(vec, word_vec[i])
        if dist < min:
            idx = i
            min = dist
    return idx

def decoder(learned_embeddings, model, word_to_ix):
  sentence = []
  sent = []
  for i in range(7023):
    for j in range(64):
      sent.append(closest(learned_embeddings[i][j], word_to_ix)[0][0])
    sentence.append(sent)
    sent = []
  return sentence


if __name__ == "__main__":
    # Loading model
    # TODO: if running on GPU just remove the map location argument
    model = torch.load('model.pt', map_location='cpu')
    losses = torch.load('loss.pt', map_location='cpu')
    word_to_ix = torch.load('idx.pt', map_location='cpu')
    recon = torch.load('tweets_recon.pt', map_location='cpu')
    
    decoded_sent = decode(recon, model, word_to_ix)
    torch.save(decoded_sent, 'decoded_sent.pt')

