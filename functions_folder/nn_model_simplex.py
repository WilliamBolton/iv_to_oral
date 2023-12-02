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

class Model_simplex(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, hid_dim2, hid_dim3, hid_dim4, dropout):
        super().__init__()

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
            nn.Dropout(dropout))
        self.layers2 = nn.Sequential(nn.Linear(hid_dim4, output_dim))

    def forward(self, x):
        
        x1 = self.layers(x)
        x1 = self.layers2(x1)

        return x1
    
    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        
        x1 = self.layers(x)
        
        return x1

class Model_simplex_short(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, hid_dim2, dropout):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim2),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.layers2 = nn.Sequential(nn.Linear(hid_dim2, output_dim))

    def forward(self, x):
        
        x1 = self.layers(x)
        x1 = self.layers2(x1)

        return x1
    
    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        
        x1 = self.layers(x)
        
        return x1

class Model_simplex_long(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, hid_dim2,  hid_dim3, dropout):
        super().__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim2, hid_dim3),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.layers2 = nn.Sequential(nn.Linear(hid_dim3, output_dim))

    def forward(self, x):
        
        x1 = self.layers(x)
        x1 = self.layers2(x1)

        return x1
    
    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        
        x1 = self.layers(x)
        
        return x1
