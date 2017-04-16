from tqdm import *
import pandas as pd
import numpy as np
import tensorflow as tf

# Constants
BATCH_SIZE = 16

# Helper functions
def create_weights(shape, dtype=None, name=None):

    if dtype is not None: start_value = tf.truncated_normal(shape=shape, dtype=dtype)
    else: start_value = tf.truncated_normal(shape=shape)

    if name is not None: weights = tf.Variable(start_value, name=name)
    else: weights = tf.Variable(start_value)

    return weights


def conv_block(input_tensor, input_shape, channels_out, is_training, name=None):

    def _convolution(input_tensor, channels_in, channels_out, name=None):

        if name is not None: conv = tf.nn.conv2d(input_tensor,
                create_weights(shape=(3,3, channels_in, channels_out), name=name+"_conv_weights"),
                strides=[1,1,1,1],
                padding='SAME',
                name=name)
        else: conv = tf.nn.conv2d(input_tensor,
                create_weights(shape=(3,3, channels_in, channels_out)),
                strides=[1,1,1,1],
                padding='SAME')

        bias = create_weights(shape=[channels_out], name=name)
        return conv + bias
    
    
    def _batch_norm(input_tensor, is_training):

        return tf.contrib.layers.batch_norm(input_tensor, is_training=is_training)


    if name is None: name = ""
    conv1 = _convolution(input_tensor, input_shape[-1], channels_out, name=name+"_conv1")
    conv2 = _convolution(conv1, channels_out, channels_out, name=name+"_conv2")

    conv2_BN = _batch_norm(conv2, is_training)
    pool = tf.nn.max_pool(conv2_BN, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    return pool


def dense_layer(input_tensor, input_shape, output_shape, name=None):
    
    # Reduce tensor to have width and height of 1 with max pool layer
    if (len(input_tensor.get_shape()) == 4):
        width = input_tensor.get_shape()[1]
        height = input_tensor.get_shape()[2]
        reduced_input = tf.nn.max_pool(input_tensor,
                ksize=[1, width, height, 1],
                strides=[1, width, height, 1],
                padding='VALID')
    else:
        reduced_input = input_tensor

    # Define weights
    weights = create_weights(shape=[output_shape, input_shape])
    bias = create_weights(shape=[output_shape])
    
    # Matrix multiplication + bias + relu
    reduced_input = tf.squeeze(reduced_input)
    logits = tf.transpose(tf.matmul(weights, tf.transpose(reduced_input)) + tf.expand_dims(bias, 1))
    return tf.nn.relu(logits)


# Define CNN
tf.reset_default_graph()

# Input
image_batch = tf.placeholder(tf.float32, name="input_batch", shape=(BATCH_SIZE, 640, 360, 3))
is_training = tf.placeholder(tf.bool, name="is_training")

cnvblk1 = conv_block(image_batch, (BATCH_SIZE, 640, 360, 3), 8, is_training=is_training, name="1")
cnvblk2 = conv_block(cnvblk1, (BATCH_SIZE, 320, 180, 8), 16, is_training=is_training, name="2")
cnvblk3 = conv_block(cnvblk2, (BATCH_SIZE, 160, 90, 16), 32, is_training=is_training, name="3")
cnvblk4 = conv_block(cnvblk3, (BATCH_SIZE, 80, 45, 32), 64, is_training=is_training, name="4")
cnvblk5 = conv_block(cnvblk4, (BATCH_SIZE, 40, 22, 64), 128, is_training=is_training, name="5")
cnvblk6 = conv_block(cnvblk5, (BATCH_SIZE, 20, 11, 128), 256, is_training=is_training, name="6")
cnvblk7 = conv_block(cnvblk6, (BATCH_SIZE, 10, 5, 256), 512, is_training=is_training, name="7")
cnvblk8 = conv_block(cnvblk7, (BATCH_SIZE, 5, 2, 512), 1024, is_training=is_training, name="8")

fc1 = dense_layer(cnvblk8, 1024, 500)
fc2 = dense_layer(fc1, 500, 8) # check the number of labels
result = tf.nn.softmax(fc2)

# Define loss funciton
