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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# Train
def train(model, dataloader, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    batch_prediction_list = []
    batch_label_list = []

    for batch_idx, sample in enumerate(tqdm(dataloader)):
        #print('batch_idx:', batch_idx)
        labels, features = sample
        features = features.float()
        labels = labels.float()
        features = features.to(device=device)
        labels = labels.to(device=device)

        # zero the gradients calculated from the last batch
        optimizer.zero_grad()

        features2 = torch.permute(features, (1, 0))
        labels2 = torch.permute(labels, (1, 0))


        # Run model
        output = model(features2)
        
        loss = criterion(output, labels2)

        # calculate the gradients
        loss.backward()

        # update the parameters of our model by doing an optimizer step
        optimizer.step()

        epoch_loss += loss.item()

        sig = torch.nn.Sigmoid()
        output = sig(output)      
        np_predictions = output.cpu().detach().numpy()
        np_labels = labels2.cpu().detach().numpy()

        np_predictions = np_predictions.squeeze()
        np_labels = np_labels.squeeze()

        np_predictions = np_predictions.flatten()
        np_labels = np_labels.flatten()
        
        # Create list
        for x in np_predictions:
            batch_prediction_list.append(x)
        for x in np_labels:
            batch_label_list.append(x)

    final_predictions = np.array(batch_prediction_list)

    final_labels = np.array(batch_label_list)

    try:
        accuracy = accuracy_score(final_labels, final_predictions.round())
    except:
        accuracy = np.nan

    try:
        auroc = roc_auc_score(final_labels, final_predictions.round())
    except:
        auroc = np.nan
    
    try:
        final_loss = epoch_loss / len(dataloader)
    except:
        final_loss = np.nan

    return final_loss, accuracy, auroc, final_predictions, final_labels
