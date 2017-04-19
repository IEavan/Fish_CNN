from tqdm import *
import random
import pandas as pd
import numpy as np
import scipy.ndimage
import scipy.misc
import os
import tensorflow as tf

# Constants
BATCH_SIZE = 4
epsilon = 1e-6
image_names = {
        "OTHER": [],
        "LAG": [],
        "SHARK": [],
        "NoF": [],
        "BET": [],
        "ALB": [],
        "YFT": [],
        "DOL": []
}

for key in image_names.keys():
    directory_path = "train/" + key
    image_names[key] = os.listdir(directory_path)

# Helper functions

# Image functions
def load_images(shape, one_hot_encoding=True):  # Shape should be (batch, height, width, channels)
    assert len(shape) is 4

    images_list = []
    labels_list = []
    for i in range(shape[0]):  # Loop over the size of the batch_size
        type_index = int(random.random() * len(image_names))
        fish_type = list(image_names.keys())[type_index]
        image_index = int(random.random() * len(image_names[fish_type]))
        raw_image = scipy.ndimage.imread("train/" + fish_type + "/" + image_names[fish_type][image_index])
        resized_image = scipy.misc.imresize(raw_image, (shape[1], shape[2]))

        if one_hot_encoding:
            one_hot_vector = np.zeros((len(image_names)))
            one_hot_vector[type_index] = 1
            labels_list.append(one_hot_vector)
        else:
            labels_list.append(fish_type)

        images_list.append(resized_image)

    
    return np.asarray(images_list), np.asarray(labels_list)


# Compute Graph functions
def create_weights(shape, dtype=tf.float32, name="weight"):

    start_value = abs(np.random.normal(0.1, 0.3, shape))
    weights = tf.Variable(start_value, name=name, dtype=dtype)
    return weights


def conv_block(input_tensor, input_shape, channels_out, is_training, name="conv_block"):

    def _convolution(_input_tensor, channels_in, channels_out, name="convolution"):

        conv = tf.nn.conv2d(_input_tensor,
                create_weights((3,3, channels_in, channels_out), name=name+"_conv_weights"),
                strides=[1,1,1,1],
                padding='SAME',
                name=name)

        bias = create_weights((channels_out), name=name+"_bias_weights")
        return conv + bias
    
    
    def _batch_norm(input_tensor, is_training):

        return tf.contrib.layers.batch_norm(input_tensor, is_training=is_training)


    conv1 = _convolution(input_tensor, input_shape[-1], channels_out, name=name+"_conv1")
    conv2 = _convolution(conv1, channels_out, channels_out, name=name+"_conv2")

    conv2_BN = _batch_norm(conv2, is_training)
    relu = tf.nn.relu(conv2_BN)
    pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

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
        reduced_input = tf.squeeze(reduced_input)
    else:
        reduced_input = input_tensor

    # Define weights
    weights = create_weights(shape=[output_shape, input_shape])
    bias = create_weights(shape=[output_shape])
    
    # Matrix multiplication + bias + relu
    logits = tf.transpose(tf.matmul(weights, tf.transpose(reduced_input)) + tf.expand_dims(bias, 1))
    return tf.nn.relu(logits)


# Define CNN
tf.reset_default_graph()

# Input
image_batch = tf.placeholder(tf.float32, name="input_batch", shape=(BATCH_SIZE, 180, 320, 3))
labels = tf.placeholder(tf.float32, name="labels", shape=(BATCH_SIZE, 8))
is_training = tf.placeholder(tf.bool, name="is_training")

cnvblk1 = conv_block(image_batch, (BATCH_SIZE, 180, 320, 3), 8, is_training=is_training, name="2")
cnvblk2 = conv_block(cnvblk1, (BATCH_SIZE, 90, 160, 8), 16, is_training=is_training, name="3")
cnvblk3 = conv_block(cnvblk2, (BATCH_SIZE, 45, 80, 16), 32, is_training=is_training, name="4")
cnvblk4 = conv_block(cnvblk3, (BATCH_SIZE, 22, 40, 32), 64, is_training=is_training, name="5")
cnvblk5 = conv_block(cnvblk4, (BATCH_SIZE, 11, 20, 64), 128, is_training=is_training, name="6")
cnvblk6 = conv_block(cnvblk5, (BATCH_SIZE, 5, 10, 128), 256, is_training=is_training, name="7")
cnvblk7 = conv_block(cnvblk6, (BATCH_SIZE, 2, 5, 256), 512, is_training=is_training, name="8")

fc1 = dense_layer(cnvblk7, 512, 250)
fc2 = dense_layer(fc1, 250, 8)
result = tf.nn.softmax(fc2)

# Define loss funciton
cross_entropy_loss_matrix = labels * tf.log(result) + (1 - labels) * tf.log(1 - result)
cross_entropy_loss = - tf.reduce_sum(cross_entropy_loss_matrix)

# Define optimizer
optimizer = tf.train.AdamOptimizer(0.5)
train_step = optimizer.minimize(cross_entropy_loss)

# Run Graph
init_vars = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_vars)
    for i in tqdm(range(30)):
        x, y = load_images((BATCH_SIZE, 180, 320, 3))
        _, loss = sess.run([train_step, cross_entropy_loss], feed_dict={image_batch: x,
                                        labels: y,
                                        is_training: True})
        if (i + 1) % 1 is 0: 
            print("training loss at iteration " + str(i) + " is: " + str(loss))
            print(sess.run(cnvblk1, feed_dict={image_batch: x, labels: y, is_training: True}))
