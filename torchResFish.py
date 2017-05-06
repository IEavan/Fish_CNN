# Import torch functionality
import torch
import torchvision
from torch.autograd import Variable

# IO
from tqdm import *

# Image loading and arrays
import numpy as np
import scipy.ndimage
import scipy.misc
import os
import random

import matplotlib.pyplot as plt
plt.ion()

# Constants
iterations = 1  # Number of training loops

# Load pretrained ResNet18
resnet = torchvision.models.resnet18(pretrained=True)

# Freeze all parameters in model
for param in resnet.parameters():
    param.require_grad = False

# Replace final fc layer with new fc layer
resnet.fc = torch.nn.Linear(512,8)

# Load image names
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


# Image loader
def load_images(shape, one_hot_encoding=True):  # Shape should be (batch, channels, height, width)
    
    # Check that four dimensions have been given
    assert len(shape) is 4

    # Create lists of image arrays and their corresponding labels
    images_list = []
    labels_list = []

    for i in range(shape[0]):  # Loop over the size of the batch_size

        # Pick a random index from the image types
        type_index = int(random.random() * len(image_names))

        # Get the abrv. name of fish type
        fish_type = list(image_names.keys())[type_index]

        # Pick random image from dataset with given label
        image_index = int(random.random() * len(image_names[fish_type]))

        # Load the image and resize it to be the given shape
        raw_image = scipy.ndimage.imread("train/" + fish_type + "/" + image_names[fish_type][image_index])
        resized_image = scipy.misc.imresize(raw_image, (shape[1], shape[2]))

        # Convert label to one hot encoding
        if one_hot_encoding:
            one_hot_vector = np.zeros((len(image_names)))
            one_hot_vector[type_index] = 1
            labels_list.append(one_hot_vector)
        else:
            labels_list.append(fish_type)

        images_list.append(resized_image)

    
    return np.asarray(images_list), np.asarray(labels_list)

# ReTrain ResNet
for iter in tqdm(range(iterations)):

    # Load images and labels
    imgs, labels = load_images((64, 224, 224, 3))
    plt.imshow(imgs[0])
    plt.pause(1)
    imgs = np.transpose(imgs, (0,3,1,2))
    imgs = torch.from_numpy(imgs)
    imgs = Variable(imgs)
    print(imgs[0])
    print(imgs[0].data.type())
    # print(resnet(imgs.float()))  # Not enough ram to do this.. :'(
