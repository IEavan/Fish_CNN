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
use_cuda = torch.cuda.is_available()

# Load pretrained ResNet18
resnet = torchvision.models.resnet18(pretrained=True)
if use_cuda: resnet.cuda()

# Freeze all parameters in model
for param in resnet.parameters():
    param.require_grad = False

# Replace final fc layer with new fc layer
resnet.fc = torch.nn.Linear(512,8)

# Load image names
sets = ['train', 'test']

image_names = [{
        "OTHER": [],
        "LAG": [],
        "SHARK": [],
        "NoF": [],
        "BET": [],
        "ALB": [],
        "YFT": [],
        "DOL": []
} for x in sets]

for i, x in enumerate(sets):
    for key in image_names[i].keys():
        directory_path = x + "/" + key
        image_names[i][key] = os.listdir(directory_path)

# Image loader
def load_images(shape, dataset="train", one_hot_encoding=True):  # Shape should be (batch, channels, height, width)
    
    # Check that four dimensions have been given
    assert len(shape) is 4

    # Create lists of image arrays and their corresponding labels
    images_list = []
    labels_list = []

    if dataset is "train":
        index = 0
    elif dataset is "test":
        index = 1
    else:
        print("Invalid parameter for load images: " + dataset)

    for i in range(shape[0]):  # Loop over the size of the batch_size

        # Pick a random index from the image types
        type_index = int(random.random() * len(image_names[index]))

        # Get the abrv. name of fish type
        fish_type = list(image_names[index].keys())[type_index]

        # Pick random image from dataset with given label
        image_index = int(random.random() * len(image_names[index][fish_type]))

        # Load the image and resize it to be the given shape
        raw_image = scipy.ndimage.imread(
                dataset + "/" + fish_type + "/" + 
                image_names[index][fish_type][image_index])
        resized_image = scipy.misc.imresize(raw_image, (shape[1], shape[2]))

        # Convert label to one hot encoding
        if one_hot_encoding:
            one_hot_vector = np.zeros((len(image_names[index])))
            one_hot_vector[type_index] = 1
            labels_list.append(one_hot_vector)
        else:
            labels_list.append(fish_type)

        images_list.append(resized_image)

    
    return np.asarray(images_list), np.asarray(labels_list)

# ReTrain ResNet
# for i in tqdm(range(iterations)):
# 
#     # Load images and labels
#     imgs, labels = load_images((64, 224, 224, 3), dataset="train")
#     plt.imshow(imgs[0])
#     plt.pause(1)
#     imgs = np.transpose(imgs, (0,3,1,2))
#     imgs = torch.from_numpy(imgs)
#     imgs = Variable(imgs)
#     print(imgs[0])
#     print(imgs[0].data.type())
#     # print(resnet(imgs.float()))  # Not enough ram to do this.. :'(

def train(model, iterations):

    params = [p for p in model.parameters() if p.requires_grad()]
    
    for i in range(iterations):

        if i % 500 is 0:
            learning_rate = 0.001 / 10 ** (i // 500)
            optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.8)

        # Forward Propagate
        imgs, labels = load_images((64, 224, 224, 3), dataset="train")
        if use_cuda: 
            imgs = Variable(torch.from_numpy(imgs).cuda())
            labels = Variable(torch.from_numpy(labels).cuda())
        else: 
            imgs = Variable(torch.from_numpy(imgs).cuda())
            labels = Variable(torch.from_numpy(labels))
        
        outputs = model(imgs.float())
        
        # Compute Loss
        loss = - (labels * outputs.log() + (1 - labels) * (1 - outputs).log()).sum()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # IO
        if i % 100 is 0:
            print("Current training loss is {}".format(loss.data[0]))

def evaluate(model, rounds=10):

    correct = 0
    total = 0

    for i in range(rounds):

        # Loaf images
        imgs, labels = load_images((64, 224, 224, 3), dataset="test")

        # Move to cuda if possible
        if use_cuda: 
            imgs = Variable(torch.from_numpy(imgs).cuda())
            labels = Variable(torch.from_numpy(labels).cuda())
        else: 
            imgs = Variable(torch.from_numpy(imgs).cuda())
            labels = Variable(torch.from_numpy(labels))

        # Forward Propagate
        outputs = model(imgs)
        _, predictions = outputs.data.max(1)
        _, label_index = labels.data.max(1)

        # Count Correct
        correct += (labels_index == predictions).sum()
        total += predictions.size()

    # Compute Accuracy
    acc = correct / total
    return acc

train(resnet, 1000)
print("Final testset accuracy is {:.2%}".format(evaluate(resnet)))
