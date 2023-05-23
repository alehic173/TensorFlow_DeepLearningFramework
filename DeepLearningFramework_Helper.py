

# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:19:05 2023

@author: ALex
Code inspired by Coursera Deep Learning Course
"""

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time

# Need tensorflow version 2.3.0



def tensor_normalize(image):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, )
    and normalize its components.
    
    Arguments
    image - Tensor.
    
    Returns: 
    result -- Transformed tensor 
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1,])
    return image

def linear_function():
    """
    Implements a linear function: 
    Returns: 
    result -- Y = WX + b 
    """

    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.add(tf.matmul(W, X), b)
    
    return Y

def sigmoid(z):
    
    """
    Computes the sigmoid of z

    """
    # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.

    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    
    # YOUR CODE ENDS HERE
    return a


# functiion takes one label and total number of classes and returns the one hot envoding in a column wise matrix
def one_hot_matrix(label, depth=6):
    """
    Computes the one hot encoding for a single label
    
    Arguments:
        label --  (int) Categorical labels
        depth --  (int) Number of different classes that label can take
    
    Returns:
         one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """

    one_hot = tf.reshape(tf.one_hot(label, depth, None), shape=[-1, ])

    return one_hot


# GRADED FUNCTION: initialize_parameters

def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
                                
    initializer = tf.keras.initializers.GlorotNormal(seed=1)   
    W1 = tf.Variable(initializer(shape=([25, 12288])))
    b1 = tf.Variable(initializer(shape=([25, 1])))
    W2 = tf.Variable(initializer(shape=([12, 25])))
    b2 = tf.Variable(initializer(shape=([12, 1])))
    W3 = tf.Variable(initializer(shape=([6, 12])))
    b3 = tf.Variable(initializer(shape=([6, 1])))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.math.add(tf.linalg.matmul(W1,X),b1)                           # Z1 = np.dot(W1, X) + b1
    A1 = tf.keras.activations.relu(Z1)                          # A1 = relu(Z1)
    Z2 = tf.math.add(tf.linalg.matmul(W2,A1),b2)                             # Z2 = np.dot(W2, A1) + b2
    A2 = tf.keras.activations.relu(Z2)                           # A2 = relu(Z2)
    Z3 = tf.math.add(tf.linalg.matmul(W3,A2),b3)                           # Z3 = np.dot(W3, A2) + b3
    

    return Z3

def compute_total_loss(logits, labels):
    """
    Computes the total loss
    
    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3
    
    Returns:
    total_loss - Tensor of the total loss value
    """
    
    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(logits,labels,from_logits=True))

    return total_loss



