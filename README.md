# PyTorch models

This repository is a collection of PyTorch models and other utilities helpful for training, testing and visualizing neural networks.

# Code structure

- All the models are in the models folder.
  - `resnet.py` contains the ResNet18 and ResNet34 models
- All the utilities are in the utils folder
  - `dataloader.py` contains the custom dataloader classes. Right now it has CIFAR10. It applies transformations as well.
  - `summary.py` contains helpers for printing model summary
  - `tester.py` contains the Tester class, which takes in various parameters and is used to test a model.
  - `trainer.py` contains the Trainer class, which takes in various parameters and is used to train a model.
  - `transforms.py` contains Transforms, which can be used to augment the data. As of now, being used by the Dataloader.
  - `utils.py` contains miscellaneous functions to plot images and graphs.
