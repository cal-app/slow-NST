import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image

import numpy as np
import tensorflow as tf

def gram_matrix(A):
    """
    The Gram matrix of a matrix A is A.A^T
    
    Usage:
    For the unrolled activation A of a layer, of shape (n_C, n_H*n_W),
    the Gram matrix basically computes the local correlations of the channels 
    
    Returns:
    Gram matrix of A, of shape (n_C, n_C)
    """
    
    gram = tf.matmul(A, tf.transpose(A))
    
    return gram
    

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Unroll each channel of the activations
    a_C_unrolled = tf.reshape(a_C, [1, n_C, n_H*n_W])
    a_G_unrolled = tf.reshape(a_G, [1, n_C, n_H*n_W])
    
    # Content cost according to eq () of the paper
    J_content = 1/(4*n_H*n_W*n_C) * tf.reduce_sum(tf.square(a_C - a_G))

    
    return J_content

def compute_layer_style_cost(a_S, a_G, content_mask = None):
    """
    Computes for style cost for a single layer
    
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    content_mask -- if provided, will apply spatial control
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    if content_mask:
        # Downsample mask size to conv output size
        mask_resized = tf.image.resize_bilinear(content_mask, [n_H, n_W], align_corners=False, name=None)

        # B&W image, so we don't need 3 channels
        mask_resized = mask_resized[:,:,:,0]

        # Add an extra dim to allow broadcasting
        mask_resized = tf.expand_dims(mask_resized, [-1])
        
        # Apply mask onto the generated output. 
        #This basically selects which parts of the image will contribute to the style cost function
        a_G = tf.math.multiply(a_G, mask_resized)
    
    # Unroll each channel of the activations (n_C, n_H*n_W) 
    a_S = tf.reshape(tf.transpose(a_S), [n_C, n_H*n_W])
    a_G = tf.reshape(tf.transpose(a_G), [n_C, n_H*n_W])

    # Compute gram matrices of the activations
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the layer style loss following equation () of ()
    J_style_layer = 1/(4 * n_C**2 * (n_H*n_W)**2) * tf.reduce_sum(tf.square(GS - GG))

    return J_style_layer
    
    
def compute_style_cost(sess, model, STYLE_LAYERS, content_mask = None):
    """
    Computes the total style cost for the layers in STYLE_LAYERS, weighted accordingly
    
    Arguments:
    sess-- the tensorflow session
    model -- VGG19, with the style image already assigned as input
    STYLE_LAYERS -- A list of (layer, weighting) tuples
    content_mask -- if provided, will apply spatial control
    
    Returns: 
    J_style -- A scalar tensor, eqn () in ()
    """
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the layer in the model
        out = model[layer_name]

        # a_S is the layer's activation for the style image, computed by running the session on the out tensor
        # This means you must run sess.run(model['input'].assign(style_image)) before calling this function
        a_S = sess.run(out)

        # a_G is the layer's activation for the generated image
        # We do not evaluate it yet, only later when the generated image is assigned and iterated over
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G, content_mask)

        # Rescale by coefficient
        J_style += coeff * J_style_layer

    return J_style
    
def total_cost(J_content, J_style, alpha, beta):
    """
    Computes the total cost function
    """

    J = alpha * J_content + beta * J_style
    
    return J