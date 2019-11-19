import sys
import torch
import itertools
import math
import random

"""Parameters"""
is_training = True
# MAX_FACES = 10
learning_rate = 0.1
num_epochs = 10
batch_size = 10
dim = (300, 300)
MODEL_PATH = r"C:\Maxim\Repositories\AI\Models"
DATASET_PATH = r"C:\Maxim\Repositories\AI\Datasets"
variances = [0.1, 0.2]


