# Self Pruning Neural Network

This project is about building a neural network that can reduce its own complexity while training. 
It uses gate parameters along with L1 regularization to remove unnecessary weights and improve efficiency.

## Features
- Implemented a custom PrunableLinear layer
- Applied sparsity regularization
- Trained on CIFAR-10 dataset
- Compared model accuracy with sparsity levels

## Run

pip install torch torchvision matplotlib
python main.py
