# U-Net Autoencoder for Impact Echo Analysis



## Overview
This project provides a framework for training and evaluating the U-Net Autoencoder. Below is a brief description of the key components included in this repository.

## Files and Descriptions

### train_class.py
- Contains the base class for all models defined in `models.py`.
- Models in `models.py` inherit from this base class.
- Implements the training process for the models.

### models.py
- Contains definitions of various machine learning models.
- Each model inherits from the base class in `train_class.py`.

### dataset.py
- Creates and manages the dataset class.
- Handles data loading and preprocessing tasks.

### detection.py
- Provides evaluation code for the trained models.
- Generates confusion matrices and plots for different test sets (note: test sets are not included in this repository).

## Usage
1. Define your models in `models.py` inheriting from the base class in `train_class.py`.
2. Use `dataset.py` to create and preprocess your datasets.
3. Train your models using the training process defined in `train_class.py`.
4. Evaluate your models with `detection.py` to generate confusion matrices and plots.

## Notes
- The test sets required for evaluation are not included in this repository. You will need to provide your own test data.
