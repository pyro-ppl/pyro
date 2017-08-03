#!/usr/bin/env python

# Char RNN + independently sampled softmax alpha for each text chunk

# Char RNN loosely based on
# https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb

# TODO:
# - use batches
# - train on gpu

import json
import math
import numpy as np
import os
import pyro
import random
import re
import string
import time
import torch
import unidecode
import visdom

import torch.nn as nn

from pprint import pprint
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import softmax, softplus

from pyro import sample, observe, param
from pyro.distributions import DiagNormal, Bernoulli, Categorical, Exponential, Delta
from pyro.infer.kl_qp import KL_QP
from pyro.poutine import block


# --------------------------------------------------------------------
# Data

class DataSet(object):
    """
    Abstract base class for text datasets
    """

    def __init__(self):
        self.text = None
        self.all_chars = string.printable
        self.test_prefixes = []

    def random_chunk(self, chunk_len=400):
        assert self.text
        assert self.text_len
        start_index = random.randint(0, self.text_len - chunk_len)
        end_index = start_index + chunk_len + 1
        return self.text[start_index:end_index]

    def char_tensor(self, string):
        assert self.all_chars
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = self.all_chars.index(string[c])
        return Variable(tensor)

    def read(self):
        pass


class StringData(DataSet):
    """
    A text dataset constructed from a given string.
    """

    def __init__(self, text):
        super(StringData, self).__init__()
        self.text = text
        self.text_len = len(text)
        self.all_chars = list(set(text))
        self.test_prefixes = [self.text[:n] for n in [5, 7, 9]]  # FIXME: generalize

    def random_chunk(self, chunk_len=400):
        if chunk_len > len(self.text):
            return self.text
        else:
            return super(StringDataSet, self).random_chunk(chunk_len=chunk_len)


class SimpleWiki(DataSet):
    """
    Assumes data is JSON output from https://github.com/attardi/wikiextractor
    using Simple English Wikipedia dumps.

    TODO: generalize (scan directory to determine how many files there are)
    """

    def __init__(self, path):
        super(SimpleWiki, self).__init__()
        self.test_prefixes = ["This is", "But he", "Why"]
        self.path = path

    def read(self):
        print("Reading SimpleWiki...")
        articles = []
        def read_articles(filename):
            with open(filename) as handle:
                text = handle.read()
                blobs = text.strip().split("\n")
                for blob in blobs:
                    obj = json.loads(blob)
                    text = obj["text"]
                    text = ''.join(s for s in text if s in self.all_chars)
                    articles.append(text)
        for i in range(0, 100):
            read_articles(os.path.join(self.path, "AA/wiki_{:02d}".format(i)))
        for i in range(0, 15):
            read_articles(os.path.join(self.path, "AB/wiki_{:02d}".format(i)))
        self.text = "\n\n\n\n".join(articles)
        self.text_len = len(self.text)
        print("Done, read {0} chars.".format(self.text_len))


class Noisy(DataSet):
    """
    Creates a noised-up version of a given dataset. For each chunk, we independently
    choose a noise probability, and then with that probability replace each character
    with a uniformly drawn character.
    """

    def __init__(self, dataset, max_noise=0.5):
        super(Noisy, self).__init__()
        self.dataset = dataset
        self.max_noise = max_noise

    def flip(self, p):
        return random.random() <= p
    
    def random_chunk(self, chunk_len=400):
        chunk = self.dataset.random_chunk(chunk_len=chunk_len)
        noise_prob = random.random() * self.max_noise
        noisy_chunk = ""
        for char in chunk:
            if self.flip(noise_prob):
                noisy_chunk += random.choice(self.all_chars)
            else:
                noisy_chunk += char
        print(noisy_chunk)
        return noisy_chunk

    def read(self):
        self.dataset.read()
        self.text = self.dataset.text
        self.text_len = self.dataset.text_len
        self.test_prefixes = self.dataset.test_prefixes
        self.all_chars = self.dataset.all_chars        
    
    def char_tensor(self, string):
        return self.dataset.char_tensor(string)


# --------------------------------------------------------------------
# Plotting

class LinePlot(object):

    def __init__(self, title, vis):
        self.vis = vis
        self.ys = []
        self.win = self.vis.line(np.array([1]), opts=dict({"title": title}))

    def update(self, y):
        self.ys.append(y)
        self.vis.updateTrace(X = np.arange(len(self.ys)), Y=np.array(self.ys), append=False, win=self.win)

    def update_from_param(self, param_name):
        self.update(param(param_name).data.cpu()[0])  # TODO: generalize to gpu / non-scalar params


# --------------------------------------------------------------------
# Char RNN model

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))


class CharRNN(nn.Module):

    def __init__(self, all_chars, char_tensor, hidden_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.all_chars = all_chars
        self.char_tensor = char_tensor
        num_chars = len(all_chars)
        self.rnn = RNN(num_chars, hidden_size, num_chars, num_layers=num_layers)

    def forward(self, prefix, alpha, temperature=0.8, condition=False, generate=0):
        assert len(prefix) > 0

        hidden = self.rnn.init_hidden()
        out = self.char_tensor(prefix[0])
        generated_chars = prefix[0]
        
        for i in range(1, len(prefix) + generate):
            out, hidden = self.rnn(out, hidden)
            ps = softmax(out.mul(alpha.expand(out.size())))
            dist = Categorical(ps, one_hot=False)
            name = 'char_{0}'.format(i)
            if i < len(prefix):
                # Use character provided in prefix
                char = prefix[i]
                if condition:
                    char_index = self.all_chars.index(char)
                    observe(name, dist, Variable(Tensor([char_index])))
            else:
                # Sample a character
                char_index = sample(name, dist).data[0][0]  # FIXME
                char = self.all_chars[char_index]            
            generated_chars += char
            out = self.char_tensor(char)

        return generated_chars


# --------------------------------------------------------------------
# Inference & runner

def main():

    # Load data
    # dataset = SimpleWiki("./data/raw/simple-wiki")
    # dataset = StringData("This is a simple test-sentence and this is the second part.")
    dataset = Noisy(StringData("This is a simple test-sentence and this is the second part."))
    dataset.read()

    # Set up viz
    vis = visdom.Visdom(env='char-rnn')
    elbo_pane = LinePlot("ELBo", vis)
    alpha_pane = LinePlot("Softmax alpha (delta guide)", vis)
    text_pane = vis.text('Hello, world!')

    # Set up char rnn model and guide
    pt_char_rnn = CharRNN(dataset.all_chars, dataset.char_tensor, 100)

    def core(prefix, alpha, condition, generate):
        char_rnn = pyro.module("char-rnn", pt_char_rnn)
        return char_rnn.forward(prefix, alpha, condition=condition, generate=generate)

    def model(prefix, condition=False, generate=0, alpha=None):
        if alpha is None:
            alpha = sample("alpha", Exponential(lam=Variable(Tensor([1]))))
        else:
            alpha = Variable(Tensor([alpha]))
        return core(prefix, alpha, condition, generate)
    
    def guide(prefix, condition=False, generate=0, alpha=None):
        if alpha is None:
            alpha_point_estimate = softplus(param("alpha-point-estimate", Variable(torch.ones(1), requires_grad=True)))
            alpha = sample("alpha", Delta(v=alpha_point_estimate))
        else:
            alpha = Variable(Tensor([alpha]))
        return core(prefix, alpha, condition, generate)

    # Set up neural net parameter inference
    optimizer = pyro.optim(torch.optim.Adam, { "lr": .005, "betas": (0.97, 0.999) })
    infer = KL_QP(model, guide, optimizer)

    # Set up softmax alpha inference
    alpha_optimizer = pyro.optim(torch.optim.SGD, { "lr": .005, "momentum": 0.1 })
    char_names = ["char_{0}".format(i) for i in range(1000)]
    alpha_model = block(model, expose=["alpha", "alpha-point-estimate"] + char_names)
    alpha_guide = block(guide, expose=["alpha", "alpha-point-estimate"] + char_names)
    alpha_infer = KL_QP(alpha_model, alpha_guide, alpha_optimizer)

    for k in range(10000000):

        # Draw a random text sample
        chunk = dataset.random_chunk(chunk_len=200)

        # Fit alpha to current text sample, keeping neural net params fixed
        prev_alpha_loss = float("-inf")
        for j in range(1000):
            alpha_loss = -alpha_infer.step(chunk, condition=True)
            alpha_pane.update_from_param("alpha-point-estimate")
            diff = abs(alpha_loss - prev_alpha_loss)
            if diff < 0.05:
                break
            prev_alpha_loss = alpha_loss
        
        # Fit neural net params (and alpha)
        loss = -infer.step(chunk, condition=True)
        
        # Update viz
        elbo_pane.update(loss)
        model_predictions = [model(prefix, generate=100, alpha=alpha) for prefix in dataset.test_prefixes for alpha in [1, 10, 100]]
        html = "<ul>{0}</ul>".format("\n".join(["<li>{0}</li>".format(prediction) for prediction in model_predictions]))
        vis.text(html, win=text_pane)


if __name__ == "__main__":
    main()
