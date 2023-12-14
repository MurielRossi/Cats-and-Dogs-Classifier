# Cats-and-Dogs-Classifier

This repository contains a binary image classifier trained to distinguish between cats and dogs using the "Cats and Dogs Dataset." The implementation includes modules for data manipulation, dataset splitting, and inference. The model is designed to perform binary classification, making it a useful tool for identifying whether an input image features a cat or a dog.

### Files Included

## Model Implementation Module (model.py):

Implements necessary data transformations and preprocessing for the model.
Defines functions to load and transform image data.
Implements the binary classification model architecture.
Defines functions for training and testing the model.


## Dataset Splitting Module (split_dataset.py):

Contains functions to split the dataset into training and testing sets.
Utilizes the "Cats and Dogs Dataset" for training the model.


## Inference Module (inference.py):

Implements functions for making predictions on new images using the trained model.


## Main Script (main_train.py):

It is the file that orchestrates all the modules. 
It takes care of making the various function calls in order to train and save the model.


## Making Inference (test_inference.py):

This file is used to carry out the inference process on each element present in the "images" folder. 
Prints both the predicted percentages for each class, both the predicted class and the ground thruth. 
Warning: the ground thruth is taken via regex from the file name. 
If the name does not contain the words "Cat" or "Dog", the Ground Thruth will be void.


### Usage
##Training the Model
To train the model, run the "main_train.py" script, which uses the modules for data manipulation and model implementation. This script will save the trained model for later use.

## Making Predictions
To make predictions on new images using the trained model, insert the images you want to predict into the "images" folder and run the "test_inference.py" file. 
The various predictions will be printed with the relative percentages. 
If you want to test the model, remember to include the name of the animal's class in the image title (so you can compare the prediction with the Ground Thruth).
Otherwise, if you're just trying to figure out whether the bizarre animal in your photo is a cat or a dog you're free to call your image "pluto."

## Dataset
The model is trained on the "Cats and Dogs Dataset," a widely used dataset for binary image classification tasks. The dataset contains labeled images of cats and dogs, making it suitable for training and evaluating the model's performance.







