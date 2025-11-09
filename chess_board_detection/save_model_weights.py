"""
Quick save for trained model weights
"""
import os
import pickle
import numpy as np
import cv2
import pandas as pd
from nn_models import CNN
import tensorflow as tf
from major_2 import resize_and_pad

# Define class mapping
piece_names = ["bing", "che", "ma", "pao", "xiang", "shi", "jiang"]
colors = ["r", "g"]
valid_classes = [f"{piece}_{color}" for piece in piece_names for color in colors]
valid_classes.append("empty")
num_classes = len(valid_classes)

# Build model architecture (CNN 2)
model_params = {
    'params': {
        'cnn': [
            {'filters': 8, 'kernel_size': (7, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
            {'filters': 32, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
            {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu'},
        ],
        'pool': [
            {'pool_size': (4, 4)},
            {'pool_size': (4, 4)},
            {'pool_size': (4, 4)},
        ],
        'output_dense': {'units': num_classes},
        'output_act': {'activation': 'softmax'},
    }
}

# Build model
model = CNN(**model_params)
model.build(input_shape=[(None, 256, 256, 3)])
model.call(np.zeros((1, 256, 256, 3), dtype=np.float32))

# Load data from training output (if available)
# If training was successful, model should already be in memory, we need to manually save weights

# To save weights, we need to re-run the last part of training, or create a separate save script

print("Model architecture created. Now you need to train it first before saving.")
print("Please run train_with_crops.py again to complete the training and saving.")


