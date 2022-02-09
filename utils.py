# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created: Nov. 9, 2019 by Sean von Bayern
# Updated:


# Import all required packages
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import datasets, models, transforms


def create_classifier(n_inputs, n_outputs, hidden_layers, dropout):

    # Shorten hidden_layers alias
    hl = hidden_layers

    # Handle edge case of empty or singlular hidden_layers lists
    if len(hl) == 0:
        layers = [("fc1", nn.Linear(n_inputs, n_outputs))]
    elif len(hl) == 1:
        layers = [("fc1", nn.Linear(n_inputs, hl[0])),
                  ("relu1", nn.ReLU()),
                  ("drop1", nn.Dropout(p=dropout)),
                  ("fc2", nn.Linear(hl[0], n_outputs))]

    # Otherwise, create classifier layers as normal
    else:

        # Create first layer
        layers = [("fc1", nn.Linear(n_inputs, hl[0])),
                  ("relu1", nn.ReLU()),
                  ("drop1", nn.Dropout(p=dropout))]

        # Create n hidden_layers between first and last layers
        for i in range(len(hl) - 1):
            layers.append(("fc" + str(i + 2), nn.Linear(hl[i], hl[i + 1])))
            layers.append(("relu" + str(i + 2), nn.ReLU()))
            layers.append(("drop" + str(i + 2), nn.Dropout(p=dropout)))

        # Create last layer
        layers.append(("fc" + str(len(hl) + 1), nn.Linear(hl[-1], n_outputs)))

    # Add output layer
    layers.append(("output", nn.LogSoftmax(dim=1)))

    # Assemble layers in classifier
    classifier = nn.Sequential(OrderedDict(layers))

    return classifier


def create_model(arch):

    # Determine n outputs from convolution layers for n inputs in classifier
    in_features = {"alexnet": 9216,
                   "densenet121": 1024,
                   "densenet161": 2208,
                   "densenet169": 1664,
                   "densenet201": 1920,
                   "resnet18": 512,
                   "resnet34": 512,
                   "resnet50": 2048,
                   "resnet101": 2048,
                   "resnet152": 2048,
                   "vgg11": 25088,
                   "vgg13": 25088,
                   "vgg16": 25088,
                   "vgg19": 25088}

    # Load requested pre-trained model
    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)
    elif arch == "densenet169":
        model = models.densenet169(pretrained=True)
    elif arch == "densenet201":
        model = models.densenet201(pretrained=True)
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
    elif arch == "resnet34":
        model = models.resnet34(pretrained=True)
    elif arch == "resnet50":
        model = models.resnet50(pretrained=True)
    elif arch == "resnet101":
        model = models.resnet101(pretrained=True)
    elif arch == "resnet152":
        model = models.resnet152(pretrained=True)
    elif arch == "vgg11":
        model = models.vgg11(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model, in_features[arch]


def create_loaders(data_dir):

    # Define image sub-directories
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    # Define transforms to match structure
    channel_norms = {"mean": [0.485, 0.456, 0.406],
                     "std": [0.229, 0.224, 0.225]}
    train = transforms.Compose([transforms.RandomRotation(30),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(0.25, 0.25, 0.25),
                                transforms.ToTensor(),
                                transforms.Normalize(**channel_norms)])
    test = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(**channel_norms)])

    # Load transformed datasets
    train_data = datasets.ImageFolder(train_dir, transform=train)
    valid_data = datasets.ImageFolder(valid_dir, transform=test)
    test_data = datasets.ImageFolder(test_dir, transform=test)

    # Define dataloaders and shuffle trainloader
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, validloader, testloader, train_data.class_to_idx


def process_image(imagepath):

    # Define transforms to match inputs
    norms = {"mean": [0.485, 0.456, 0.406],
             "std": [0.229, 0.224, 0.225]}
    process = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(**norms)])

    # Apply transformations to PIL image, incl. converting to tensor
    image = process(Image.open(imagepath))

    return image


def show_image(image, ax=None, title=None):

    # Create subplot if one isn't provided
    if ax is None:
        fig, ax = plt.subplots()

    # Re-order PyTorch tensor for MatPlotLib
    image = image.numpy().transpose((1, 2, 0))

    # Revert image processing
    norms = {"mean": [0.485, 0.456, 0.406],
             "std": [0.229, 0.224, 0.225]}
    image = np.array(norms["std"]) * image + np.array(norms["mean"])

    # Squish all values to between 0 and 1
    image = np.clip(image, 0, 1)

    # Arrange image tensor on ax
    ax.imshow(image)

    # Add title
    ax.set_title(title)

    return ax
