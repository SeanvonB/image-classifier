# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created: Nov. 9, 2019 by Sean von Bayern
# Updated:


# Import all required packages
import argparse
import json
import torch
from torch import nn
from utils import create_classifier, create_loaders, create_model


def get_args():

    # Create parser object
    parser = argparse.ArgumentParser()
    parser.add_argument("-arch", "--arch", type=str, default="alexnet",
                        help="name of pre-trained CNN model")
    parser.add_argument("-data_dir", "--data_dir", type=str, default="data",
                        help="path for image data folder")
    parser.add_argument("-save_dir", "--save_dir", type=str,
                        default="checkpoint.pth",
                        help="path for saved data file")
    parser.add_argument("-learn_rate", "--learn_rate", type=float,
                        default=0.001, help="learning rate of model")
    parser.add_argument("-hidden_layers", "--hidden_layers", type=list,
                        default=[1024, 512, 256, 128],
                        help="list of hidden classifier layers")
    parser.add_argument("-dropout", "--dropout", type=float, default=0.5,
                        help="probability of dropping node during training")
    parser.add_argument("-epochs", "--epochs", type=int, default=5,
                        help="number of full training cycles")
    parser.add_argument("-test_freq", "--test_freq", type=int, default=20,
                        help="how frequently to run validation tests")
    parser.add_argument("-gpu", "--gpu", type=bool, default=True,
                        help="use True for GPU; use False CPU")

    return parser.parse_args()


def save_model(args, model, optimizer, in_features, n_classes):

    # Change to evaluation mode
    model.eval()

    # Define checkpoint content
    checkpoint = {"arch": args.arch,
                  "class_to_idx": model.class_to_idx,
                  "classifier.state_dict": model.classifier.state_dict(),
                  "dropout": args.dropout,
                  "epochs": args.epochs,
                  "in_features": in_features,
                  "hidden_layers": args.hidden_layers,
                  "learn_rate": args.learn_rate,
                  "n_classes": n_classes,
                  "optimizer.state_dict": optimizer.state_dict()}

    # Save checkpoint
    torch.save(checkpoint, args.save_dir)


def test_model(model, criterion, testloader, device):

    # Define counters
    accuracy = 0
    testing_loss = 0

    # Print log header
    print(f"\nResults of model test:\n")

    # Change to evaluation mode
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:

            # Send data to active device
            images, labels = images.to(device), labels.to(device)

            # Run validation pass
            log_probs = model(images)
            batch_loss = criterion(log_probs, labels)
            probs = torch.exp(log_probs)
            top_class = probs.topk(1, dim=1)[1]
            equals = top_class == labels.view(*top_class.shape)

            # Increment counters
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            testing_loss += batch_loss.item()

    # Log results
    print(f"Testing Loss: {testing_loss / len(testloader):0.3f}  ",
          f"Accuracy: {accuracy / len(testloader):0.2%}")


def train_model(args, model, criterion, optimizer,
                trainloader, validloader, device):

    # Define counters
    epochs = args.epochs
    steps = 0
    test_freq = args.test_freq
    training_loss = 0

    # Change to training mode
    model.train()

    # Print log header
    print(f"\nResults of {(100 / test_freq) * epochs:0.0f} validations over {epochs} epochs:\n")

    # Begin training cycle
    for epoch in range(args.epochs):
        for images, labels in trainloader:

            # Send data to active device
            images, labels = images.to(device), labels.to(device)

            # Run training pass
            optimizer.zero_grad()
            log_probs = model(images)
            batch_loss = criterion(log_probs, labels)
            batch_loss.backward()
            optimizer.step()

            # Increment counters
            training_loss += batch_loss.item()
            steps += 1

            # Run validation according to test frequency
            if steps % test_freq == 0:
                accuracy = 0
                valid_loss = 0

                # Change to evaluation mode
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:

                        # Send data to active device
                        images, labels = images.to(device), labels.to(device)

                        # Run validation pass
                        log_probs = model(images)
                        batch_loss = criterion(log_probs, labels)
                        probs = torch.exp(log_probs)
                        top_class = probs.topk(1, dim=1)[1]
                        equals = top_class == labels.view(*top_class.shape)

                        # Increment counters
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        valid_loss += batch_loss.item()

                print(f"Epoch {epoch + 1:<2}  ",
                      f"Training Loss: {training_loss / test_freq:0.3f}  ",
                      f"Validation Loss: {valid_loss / len(validloader):0.3f}  ",
                      f"Accuracy: {accuracy / len(validloader):0.2%}")

                # Reset counters
                training_loss = 0

                # Revert to training mode
                model.train()


def main():

    # Store command line args
    args = get_args()
    arch = args.arch
    data_dir = args.data_dir
    learn_rate = args.learn_rate
    hidden_layers = args.hidden_layers
    dropout = args.dropout
    gpu = args.gpu

    # Define active device
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")

    # Create loaders
    trainloader, validloader, testloader, class_to_idx = create_loaders(data_dir)

    # Load class_to_name.json to get n_classes
    with open('class_to_name.json', 'r') as openfile:
        class_to_name = json.load(openfile)
    n_classes = len(class_to_name)

    # Define supported architectures
    supported_archs = ["alexnet",
                       "densenet121",
                       "densenet161",
                       "densenet169",
                       "densenet201",
                       "resnet18",
                       "resnet34",
                       "resnet50",
                       "resnet101",
                       "resnet152",
                       "vgg11",
                       "vgg13",
                       "vgg16",
                       "vgg19"]

    # Log warning if requested architecture isn't supported, then die
    if arch not in supported_archs:
        print(f"\nUnfortunately, '{arch}' isn't a supported arch.",
              f"\nSupported archs: {supported_archs}")
        return

    # Create model of requested architecture and retrieve in_features
    model, in_features = create_model(arch)

    # Add class_to_idx to model
    model.class_to_idx = class_to_idx

    # Add classifier to model
    model.classifier = create_classifier(in_features, n_classes,
                                         hidden_layers, dropout)

    # Define loss function: NLLLoss used due to LogSoftmax
    criterion = nn.NLLLoss()

    # Send model to active device prior (!) to constructing optimizer
    model.to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learn_rate)

    # Train model
    train_model(args, model, criterion, optimizer,
                trainloader, validloader, device)

    # Test model
    test_model(model, criterion, testloader, device)

    # Save model
    save_model(args, model, optimizer, in_features, n_classes)


if __name__ == "__main__":
    main()
