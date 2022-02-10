# Image Classifier

[LIVE DEVELOPMENT NOTEBOOK](https://seanvonb.github.io/image-classifier/)

This was the final project of my AI Programming Nanodegree from Udacity, which I completed in 2019. The purpose of this project was to train an image classifier for use in a hypothetical smart phone app – in this case, an app that could identify the name of a flower simply by looking at it with the phone's camera – and bundle that solution into a distributable Python package.

As such, this project has two main components:
1.	A Jupyter Notebook (see above) documenting the initial development process of an image classifier for a hypothetical flower-detector app, including step-by-step breakdown of the actual code
2.	A Python package (`predict.py`, `train.py`, and `utils.py`) for training and deploying a flexible image classifier of any kind directly from the command line – image data not included

Completing this assignment was more so a milestone of my progress with linear algebra, statistics, and calculus than with coding. So, I'd like to acknowledge Grant Sanderson and [3Blue1Brown](https://www.3blue1brown.com/), who helped me appreciate these subjects much more than traditional education did – please subscribe to his channel if you, like me, skated through high school/college math without developing any real passion for it.

## Features

-   By default, deploy a network capable of identifying 102 flower species
-	Replace the data with your own (e.g. from [ImageNet](https://image-net.org/) or [COCO](https://cocodataset.org/#home)) and classify whatever you want
-	Use `train.py` to build new classifiers, test architectures and hyperparameters, and save checkpoints
-	Use `predict.py` to load checkpoints and make inferences as needed by an application
-	Find all available arguments in the command line with the `help` argument

## Credits

-   This project was part of my [AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).
-   Image data was provided by the University of Oxford's [Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/).

## License

Copyright © 2019-2022 Sean von Bayern  
Licensed under the [MIT License](LICENSE.md)
