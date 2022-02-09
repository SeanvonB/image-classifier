# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created: Nov. 9, 2019 by Sean von Bayern
# Updated:


# Import all required packages
import argparse
import json
import matplotlib.pyplot as plt
import torch
from utils import create_classifier, create_model, process_image, show_image


def classify_image(imagepath, model, top_k, device):

    # Process image into tensor
    image = process_image(imagepath)
    image = image.unsqueeze(0)

    # Change to evaluation mode
    model.eval()

    # Run image through model
    with torch.no_grad():

        # Send image and model to active device
        image.to(device)
        model.to(device)

        # Run image through model to get top-Ks
        log_probs = model(image)
        probs = torch.exp(log_probs)
        top_probs, top_classes = probs.topk(top_k, dim=1)

        # Convert top-K classes into label indices, and convert to lists
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        class_list = [idx_to_class[i] for i in top_classes[0].tolist()]
        probs_list = top_probs[0].tolist()

        return probs_list, class_list


def get_args():

    # Create parser object
    parser = argparse.ArgumentParser()
    parser.add_argument("-imagepath", "--imagepath", type=str,
                        default="data/test/1/image_06734.jpg",
                        help="path for image file that will be classified")
    parser.add_argument("-checkpoint", "--checkpoint", type=str,
                        default="checkpoint.pth",
                        help="path for saved data file")
    parser.add_argument("-top_k", "--top_k", type=int,
                        default=3, help="number of top predictions to show")
    parser.add_argument("-gpu", "--gpu", type=bool, default=True,
                        help="use True for GPU; use False CPU")

    return parser.parse_args()


def load_model(filepath):

    # Load checkpoint of trained model
    checkpoint = torch.load(filepath)

    # Re-build model from checkpoint
    model = create_model(checkpoint["arch"])
    model.classifier = create_classifier(checkpoint["in_features"],
                                         checkpoint["n_classes"],
                                         checkpoint["hidden_layers"],
                                         checkpoint["dropout"])
    model.classifier.load_state_dict(checkpoint["classifier.state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def main():

    # Store command line args
    args = get_args()
    imagepath = args.imagepath
    checkpoint = args.checkpoint
    top_k = args.top_k
    gpu = args.gpu

    # Define active device
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")

    # Load class_to_name.json
    with open('class_to_name.json', 'r') as openfile:
        class_to_name = json.load(openfile)

    # Load and re-build trained model
    model = load_model(checkpoint)

    # Classify image
    probs, classes = classify_image(imagepath, model, top_k, device)

    # Get image and class names
    image_name = imagepath.split("/")[-1][0:-4]
    class_names = [class_to_name[str(i)] for i in classes]

    # Create figure plot to display results
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2, nrows=1)
    ax1.barh(class_names, probs)
    ax1.invert_yaxis()
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    show_image(process_image(imagepath), ax2, image_name)


if __name__ == "__main__":
    main()
