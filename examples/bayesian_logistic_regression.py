#!/usr/bin/env python
"""Bayesian logistic regression using Hamiltonian Monte Carlo.

We visualize the fit.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pdb import set_trace as bb
from edward.models import Bernoulli, Normal, Empirical
import pandas as pd
import sklearn
from sklearn.preprocessing import normalize

#UCI dataset
fname = "data/covtype.data"
with open(fname, "r+") as f:
    content = f.read()
#     f.seek(0, 0)
#     f.write(first_line.rstrip('\r\n') + '\n' + content)
df = pd.read_csv(fname, header=None)


def build_toy_dataset(N, noise_std=0.1):
  D = 1
  X = np.linspace(-6, 6, num=N)
  y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
  y[y < 0.5] = 0
  y[y >= 0.5] = 1
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D))
  return X, y


ed.set_seed(42)

N = 581012  # number of data points
# D = 39  # number of features
D = 54  # number of features

# DATA
# X_train, y_train = build_toy_dataset(N)
data = normalize(df.as_matrix(), norm='l2', axis=1)
# X_train = df.as_matrix(columns=range(D))
X_train = data[:,:-1]
# y_train = np.squeeze(df.as_matrix(columns=[D]))
y_train = data[:,-1]
print("X shape",X_train.shape)
print (y_train.shape)

# MODEL
X = tf.placeholder(tf.float32, [N, D])
w = Normal(loc=tf.zeros([D]), scale=3.0 * tf.ones([D]))
b = Normal(loc=tf.zeros([]), scale=3.0 * tf.ones([]))
y = Bernoulli(logits=ed.dot(X, w) + b)
print('dot shape', ed.dot(X,w).shape)

# INFERENCE
# T = 5000  # number of samples
# qw = Empirical(params=tf.Variable(tf.random_normal([T, D])))
# qb = Empirical(params=tf.Variable(tf.random_normal([T])))
#
# inference = ed.HMC({w: qw, b: qb}, data={X: X_train, y: y_train})
# inference.initialize(n_print=10, step_size=0.6)

# Alternatively, use variational inference.
qw_loc = tf.Variable(tf.random_normal([D]))
qw_scale = tf.nn.softplus(tf.Variable(tf.random_normal([D])))
qb_loc = tf.Variable(tf.random_normal([]) + 10)
# bb()
qb_scale = tf.nn.softplus(tf.Variable(tf.random_normal([])))

qw = Normal(loc=qw_loc, scale=qw_scale)
qb = Normal(loc=qb_loc, scale=qb_scale)
q = tf.Print(qw, [qw_loc, qw_scale], message="HERE: ")

inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
inference.initialize(n_print=10, n_iter=50)

tf.global_variables_initializer().run()

# Set up figure.
# fig = plt.figure(figsize=(20, 20), facecolor='white')
# ax = fig.add_subplot(111, frameon=False)
# plt.ion()
# plt.show(block=False)

# Build samples from inferred posterior.

print("qw,loc",qw_loc)
# tf.Print(qw.loc,  [qw.loc])
# print("qw scale", tf.Print(qw.scale))
# print("qb loc", tf.Print(qb.loc))
# print("qb scale", tf.Print(qb.scale))

# bb()
n_samples = 50
inputs = np.linspace(-100, 100, num=21600, dtype=np.float32).reshape((400, 54))
# inputs = np.linspace(-100, 100, num=15600, dtype=np.float32).reshape((400, 39))
# inputs = np.linspace(-5, 3, num=39, dtype=np.float32).reshape((39, 1))
# probs = tf.stack([tf.sigmoid(ed.dot(inputs, qw.sample()) + qb.sample())
#                   for _ in range(n_samples)])
# mul = ed.dot(inputs, tf.expand_dims(qw.sample(), axis=1))
mul = ed.dot(inputs,qw.sample())
dot = mul + qb.sample()
stack = [tf.sigmoid(dot) for _ in range(n_samples)]
probs = tf.stack(stack)

start = time.clock()
for t in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)
#
  if t % inference.n_print == 0:
    outputs = probs.eval()
#
    # Plot data and functions
#     plt.cla()
#     ax.plot(X_train[:], y_train, 'bx')
#     for s in range(n_samples):
#       ax.plot(inputs[:], outputs[s], alpha=0.2)
# #
#     ax.set_xlim([-5, 3])
#     ax.set_ylim([-0.5, 1.5])
#     plt.draw()
#     plt.pause(1.0 / 60.0)
end = time.clock()
print("total time: ", end-start)
