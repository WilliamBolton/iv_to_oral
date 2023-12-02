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

class MIMICDataset(Dataset):

    def __init__(self, path):
        self.mimic_df = path

    def __len__(self):
        return len(self.mimic_df)

    def __getitem__(self, idx):
        selected_rows = self.mimic_df.iloc[[idx]] # Double brackets to return df
        selected_rows.drop(['stay_id'], axis=1, inplace=True)
        labels = selected_rows[['po_flag']].to_numpy()
        features = selected_rows.drop(['po_flag'], axis=1).to_numpy()

        sample = {"labels": torch.from_numpy(labels).squeeze(0), "features": torch.from_numpy(features).squeeze(0)}
        return sample

    def collate_fn_padd(self, batch):
        # Extract contexts and actions from data list
        labels = [sample["labels"] for sample in batch]
        features = [sample["features"] for sample in batch]

        pad_key = 123456  # Key to identify padded values
        padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=False, padding_value=pad_key) # Keep to convert to tensor
        padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=False, padding_value=pad_key)
        
        return padded_labels, padded_features
