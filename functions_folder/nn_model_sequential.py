import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import time
import sklearn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

class Model_sequential(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, hid_dim2, hid_dim3, hid_dim4, dropout):
        super().__init__()

        # Define sequential
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim2),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hid_dim2, hid_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim2, hid_dim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim3, hid_dim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim3, hid_dim4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim4, output_dim))

    def forward(self, x):
        
        x1 = self.layers(x)

        return x1
