# COMP6721-Group-ResNet

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Visualization](#visualization)
7. [How to Run](#how-to-run)
8. [Acknowledgements](#acknowledgements)

## Project Overview
This project involves training a supervised decision tree model, a semi-supervised model and a Convolutional Neural Network (CNN) to classify images into different categories. The model is built using sklearn and PyTorch and is trained, validated, and tested on a dataset of images. The code includes data preprocessing, model training, validation, testing, and visualization of results.

## Dependencies
The following Python libraries are required to run the code:

- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
You can install these libraries using pip:

```
pip install torch torchvision numpy matplotlib scikit-learn
```

## Dataset
The dataset should be structured in a way that datasets.ImageFolder can read it. This means the dataset directory should contain one subdirectory per class, and each subdirectory should contain the images for that class.

You can download a sample small test dataset from this repository.

## Training
To train the model, follow these steps:
1. Place your dataset in a directory (e.g., C:\dataset).
2. Adjust the path variable in the code to point to your dataset directory.
3. Run the code to start training. The training progress, including loss and accuracy for both training and validation sets, will be printed to the console.

## Evaluation
After training, the model will be evaluated on a test set. The test loss and accuracy will be printed, and a confusion matrix and classification report will be generated to show the model's performance.

## Visualization
The code includes functionality to visualize:

- Training and validation loss over epochs
- Training and validation accuracy over epochs
- Confusion matrix

These visualizations help in understanding the model's performance and diagnosing potential issues.


## How to Run
1. Ensure you have installed all dependencies.
2. Download and place the sample dataset in the specified directory.
3. Run the provided code to train and evaluate the model.
4. Check the output for training/validation loss, accuracy, and visualizations.

## Acknowledgements
- The project uses scikit-learn for decision trees.
- The project uses PyTorch for building and training the CNN.
- The dataset loading and transformation are handled using torchvision.
