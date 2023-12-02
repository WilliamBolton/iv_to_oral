# Libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import time
import random
import sklearn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

# Remove printing error
pd.options.mode.chained_assignment = None

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

from typing import Counter
import imblearn
from imblearn.over_sampling import SMOTE

from functions_folder.nn_MIMICDataset import MIMICDataset
from functions_folder.nn_train import train
from functions_folder.nn_evaluate import evaluate
from functions_folder.nn_equalized_odds import equalised_odds
from functions_folder.nn_model_simplex import *

import shap

from fairlearn.postprocessing import ThresholdOptimizer
from functions_folder.sk_learn_model import *
from fairlearn.postprocessing import plot_threshold_optimizer
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate, equalized_odds_ratio, count
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from fairlearn.postprocessing._interpolated_thresholder import InterpolatedThresholder
from sklearn.utils import Bunch
from fairlearn.postprocessing._threshold_operation import ThresholdOperation

import scipy.stats as stats
import random
from numpy import mean
from numpy import std

# Set the random seeds for deterministic results.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

def smote_fun(train_data):
    # Split X y
    train_data_X = train_data.drop(columns=['stay_id', 'po_flag'])
    train_data_y = train_data['po_flag']
    Counter(train_data_y)
    oversample = SMOTE()
    train_data_X, train_data_y = oversample.fit_resample(train_data_X, train_data_y)
    Counter(train_data_y)
    train_data_y = pd.DataFrame(train_data_y, columns=['po_flag'])
    train_data_y['stay_id'] = 'x'
    train_data = pd.concat([train_data_y, train_data_X], axis=1)
    train_data = train_data.sample(frac=1, random_state=0).reset_index(drop=True)
    return train_data

# Function to split data so even distribution between val and test
def data_fun(data, individual, n_cv=10, smote_bool=True):
    
    data_dict = {}
    random_x_list = []
    x = -1

    for i in range(n_cv):
        x += 1
        #print('x', x)
        stays = data['stay_id'].unique()
        random.Random(x).shuffle(stays)
        X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
        # Filter for features in this individual
        X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
        model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
        model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
        n = round(0.7 * len(stays))
        n2 = round(0.85 * len(stays))
        train_stays = stays[:n]
        validation_stays = stays[n:n2]
        test_stays = stays[n2:]
        train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
        valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
        test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
        # Oversample train set
        if smote_bool == True:
            train_data = smote_fun(train_data)
        
        while not math.isclose(test_data.po_flag.value_counts(normalize=True)[1], valid_data.po_flag.value_counts(normalize=True)[1], abs_tol=0.005): # Check to make sure val and test set are comparable
            x +=1
            #print('x', x)
            stays = data['stay_id'].unique()
            random.Random(x).shuffle(stays) 
            X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
            # Filter for features in this individual
            X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
            model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
            model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
            n = round(0.7 * len(stays))
            n2 = round(0.85 * len(stays))
            train_stays = stays[:n]
            validation_stays = stays[n:n2]
            test_stays = stays[n2:]
            train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
            valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
            test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
            # Oversample train set
            if smote_bool == True:
                train_data = smote_fun(train_data)

        data_dict[i] = [train_data, valid_data, test_data]
        random_x_list.append(x)

    return data_dict, random_x_list

# Define how long an epoch takes
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# Initializing the weights of our model.
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

# Function to train and eval model 
def run_fun(data_dict, model, string):

    #overall_best_valid_auroc = 0
    overall_best_test_auroc = 0

    test_auroc_results = []
    test_accuracy_results = []
    test_balanced_accuracy_results = []
    test_recall_results = []
    test_precision_results = []
    test_f1_results = []
    test_auprc_results = []
    test_cm_results = []
    test_true_positive_rate_results = []
    test_fasle_positive_rate_results = []

    ub_test_auroc_results = []
    ub_test_accuracy_results = []
    ub_test_balanced_accuracy_results = []
    ub_test_recall_results = []
    ub_test_precision_results = []
    ub_test_f1_results = []
    ub_test_auprc_results = []
    ub_test_cm_results = []
    ub_test_true_positive_rate_results = []
    ub_test_fasle_positive_rate_results = []

    master_equalised_odds_df = pd.DataFrame()

    # Define batch size 
    batch_size = 256

    # Define optimizer and learning_rate
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss
    criterion = nn.BCEWithLogitsLoss()

    # Define epochs and clip
    N_EPOCHS = 10 #10
    CLIP = 1

    # Iterate through dict i.e fold
    for key, value in data_dict.items():
        train_data = value[0]
        valid_data = value[1]
        test_data = value[2]

        # Initializing the weights of our model each fold
        model.apply(init_weights)
        
        # Define dataloaders
        train_dataset = MIMICDataset(train_data)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn_padd)

        valid_dataset = MIMICDataset(valid_data)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn_padd)

        test_dataset = MIMICDataset(test_data)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn_padd)

        # Run
        best_valid_loss = float('inf')
        best_valid_auroc = 0

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_accuracy, train_auroc, train_predictions, train_labels = train(model, train_dataloader, optimizer, criterion, CLIP)
            valid_loss, valid_accuracy, valid_auroc, valid_predictions, valid_labels = evaluate(model, valid_dataloader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #print('BEST VALID LOSS')

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                #print('UPDATED BEST INTERMEDIATE MODEL')
                torch.save(model.state_dict(), f'hold_out_switch_model_intermediate_{string}.pt')

        # -----------------------------
        # Evaluate best model on test set
        # -----------------------------

        model.load_state_dict(torch.load(f'hold_out_switch_model_intermediate_{string}.pt'))

        test_loss, test_accuracy, test_auroc, test_predictions, test_labels = evaluate(model, test_dataloader, criterion)

        print('Test AUROC result:', test_auroc)

        # Use new cut off
        lower_bound_test_predictions, upper_bound_test_predictions = new_threshold_fun(test_predictions)

        # Lower bound
        test_auroc2 = roc_auc_score(test_labels, lower_bound_test_predictions)
        print('Test AUROC result 2:', test_auroc2)
        test_accuracy2 = accuracy_score(test_labels, lower_bound_test_predictions)
        #assert test_accuracy == test_accuracy2
        test_balanced_accuracy = balanced_accuracy_score(test_labels, lower_bound_test_predictions)
        test_recall = recall_score(test_labels, lower_bound_test_predictions)
        test_precision = precision_score(test_labels, lower_bound_test_predictions)
        test_f1 = f1_score(test_labels, lower_bound_test_predictions)
        test_auprc = average_precision_score(test_labels, lower_bound_test_predictions)
        test_cm = confusion_matrix(test_labels, lower_bound_test_predictions)
        tn, fp, fn, tp = test_cm.ravel()
        test_true_positive_rate = (tp / (tp + fn))
        test_false_positive_rate = (fp / (fp + tn))

        # Upper bound
        ub_test_auroc2 = roc_auc_score(test_labels, upper_bound_test_predictions)
        ub_test_accuracy2 = accuracy_score(test_labels, upper_bound_test_predictions)
        ub_test_balanced_accuracy = balanced_accuracy_score(test_labels, upper_bound_test_predictions)
        ub_test_recall = recall_score(test_labels, upper_bound_test_predictions)
        ub_test_precision = precision_score(test_labels, upper_bound_test_predictions)
        ub_test_f1 = f1_score(test_labels, upper_bound_test_predictions)
        ub_test_auprc = average_precision_score(test_labels, upper_bound_test_predictions)
        ub_test_cm = confusion_matrix(test_labels, upper_bound_test_predictions)
        tn, fp, fn, tp = ub_test_cm.ravel()
        ub_test_true_positive_rate = (tp / (tp + fn))
        ub_test_false_positive_rate = (fp / (fp + tn))

        # Check fairness
        equalised_odds_df = equalised_odds(test_data, batch_size, model, criterion)
        master_equalised_odds_df = pd.concat([master_equalised_odds_df, equalised_odds_df], axis=0)
        #print(master_equalised_odds_df)
        
        #confusion_matrix(test_labels.round(), test_predictions.round())

        if test_auroc2 > overall_best_test_auroc:
            overall_best_test_auroc = test_auroc2
            print('UPDATED BEST OVERALL MODEL')
            #torch.save(model.state_dict(), f'hold_out_switch_model_{string}.pt') # Hastag out when dont want to change
        
        test_auroc_results.append(test_auroc2)
        test_accuracy_results.append(test_accuracy2)
        test_balanced_accuracy_results.append(test_balanced_accuracy)
        test_recall_results.append(test_recall)
        test_precision_results.append(test_precision)
        test_f1_results.append(test_f1)
        test_auprc_results.append(test_auprc)
        test_cm_results.append(test_cm)
        test_true_positive_rate_results.append(test_true_positive_rate)
        test_fasle_positive_rate_results.append(test_false_positive_rate)

        ub_test_auroc_results.append(ub_test_auroc2)
        ub_test_accuracy_results.append(ub_test_accuracy2)
        ub_test_balanced_accuracy_results.append(ub_test_balanced_accuracy)
        ub_test_recall_results.append(ub_test_recall)
        ub_test_precision_results.append(ub_test_precision)
        ub_test_f1_results.append(ub_test_f1)
        ub_test_auprc_results.append(ub_test_auprc)
        ub_test_cm_results.append(ub_test_cm)
        ub_test_true_positive_rate_results.append(ub_test_true_positive_rate)
        ub_test_fasle_positive_rate_results.append(ub_test_false_positive_rate)

    test_results = [test_auroc_results, test_accuracy_results,
        test_balanced_accuracy_results,
        test_recall_results,
        test_precision_results,
        test_f1_results,
        test_auprc_results,
        test_cm_results,
        test_true_positive_rate_results,
        test_fasle_positive_rate_results
        ]
    
    ub_test_results = [ub_test_auroc_results, ub_test_accuracy_results,
        ub_test_balanced_accuracy_results,
        ub_test_recall_results,
        ub_test_precision_results,
        ub_test_f1_results,
        ub_test_auprc_results,
        ub_test_cm_results,
        ub_test_true_positive_rate_results,
        ub_test_fasle_positive_rate_results
        ]
    
    master_equalised_odds_df.set_index(['column', 'value'], inplace=True)
    by_row_index = master_equalised_odds_df.groupby(master_equalised_odds_df.index)
    mean_equalised_odds_df = by_row_index.mean()
    sd_equalised_odds_df = by_row_index.std()

    return test_results, ub_test_results, mean_equalised_odds_df, sd_equalised_odds_df

def new_threshold_fun(predictions, lower_bound=0.5427614, upper_bound=0.7364093): # Note changed to short thresholds
    lower_bound_predictions = [1 if a_ >= lower_bound else 0 for a_ in predictions]
    upper_bound_predictions = [1 if a_ >= upper_bound else 0 for a_ in predictions]
    return lower_bound_predictions, upper_bound_predictions

def analyze_results_fun(test_results):
    # Assign results
    test_auroc_results, test_accuracy_results,test_balanced_accuracy_results,test_recall_results,test_precision_results,test_f1_results,test_auprc_results,test_cm_results, test_tpr_results, test_fpr_results = [test_results[i] for i in range(len(test_results))]
    print('mean test_auroc:', np.array(test_auroc_results).mean())
    print('std test_auroc:', np.array(test_auroc_results).std())
    print('test_auroc 2.5th percentile:', max(0, np.percentile(test_auroc_results, 2.5)))
    print('test_auroc 97.5th percentile:', min(1, np.percentile(test_auroc_results, 97.5)))
    print('mean test_accuracy:', np.array(test_accuracy_results).mean())
    print('std test_accuracy:', np.array(test_accuracy_results).std())
    print('test_accuracy 2.5th percentile:', max(0, np.percentile(test_accuracy_results, 2.5)))
    print('test_accuracy 97.5th percentile:', min(1, np.percentile(test_accuracy_results, 97.5)))
    print('mean test_balanced_accuracy:', np.array(test_balanced_accuracy_results).mean())
    print('std test_balanced_accuracy:', np.array(test_balanced_accuracy_results).std())
    print('test_balanced_accuracy 2.5th percentile:', max(0, np.percentile(test_balanced_accuracy_results, 2.5)))
    print('test_balanced_accuracy 97.5th percentile:', min(1, np.percentile(test_balanced_accuracy_results, 97.5)))
    print('mean test_recall:', np.array(test_recall_results).mean())
    print('std test_recall:', np.array(test_recall_results).std())
    print('test_recall 2.5th percentile:', max(0, np.percentile(test_recall_results, 2.5)))
    print('test_recall 97.5th percentile:', min(1, np.percentile(test_recall_results, 97.5)))
    print('mean test_precision:', np.array(test_precision_results).mean())
    print('std test_precision:', np.array(test_precision_results).std())
    print('test_precision 2.5th percentile:', max(0, np.percentile(test_precision_results, 2.5)))
    print('test_precision 97.5th percentile:', min(1, np.percentile(test_precision_results, 97.5)))
    print('mean test_f1:', np.array(test_f1_results).mean())
    print('std test_f1:', np.array(test_f1_results).std())
    print('test_f1 2.5th percentile:', max(0, np.percentile(test_f1_results, 2.5)))
    print('test_f1 97.5th percentile:', min(1, np.percentile(test_f1_results, 97.5)))
    print('mean test_auprc:', np.array(test_auprc_results).mean())
    print('std test_auprc:', np.array(test_auprc_results).std())
    print('test_auprc 2.5th percentile:', max(0, np.percentile(test_auprc_results, 2.5)))
    print('test_auprc 97.5th percentile:', min(1, np.percentile(test_auprc_results, 97.5)))
    print('mean test_tpr:', np.array(test_tpr_results).mean())
    print('std test_tpr:', np.array(test_tpr_results).std())
    print('test_tpr 2.5th percentile:', max(0, np.percentile(test_tpr_results, 2.5)))
    print('test_tpr 97.5th percentile:', min(1, np.percentile(test_tpr_results, 97.5)))
    print('mean test_fpr:', np.array(test_fpr_results).mean())
    print('std test_fpr:', np.array(test_fpr_results).std())
    print('test_fpr 2.5th percentile:', max(0, np.percentile(test_fpr_results, 2.5)))
    print('test_fpr 97.5th percentile:', min(1, np.percentile(test_fpr_results, 97.5)))

# Calculate the number of trainable parameters in the model.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fairness_fun(data, train_data, test_data, string='insurance', constraint="equalized_odds", cv=False):

  demographics = demographics_fun(data) 

  # Filter for insurance
  demographics = demographics[['stay_id', string]]

  # Change column to type string not object ...
  demographics[string] = pd.Series(demographics[string], dtype="string")

  # Get data
  new_train_data = pd.merge(train_data, demographics)
  new_train_data_x = new_train_data.drop(columns=['stay_id', 'po_flag', string])
  new_train_data_y = new_train_data['po_flag']
  new_train_data_sens = new_train_data[[string]]

  new_test_data = pd.merge(test_data, demographics)
  new_test_data_x = new_test_data.drop(columns=['stay_id', 'po_flag', string])
  new_test_data_y = new_test_data['po_flag']
  new_test_data_sens = new_test_data[[string]]

  # Fair model
  if cv == True:
    sk_model = sk_model_simplex_short()
  else:
    sk_model = sk_model_simplex_short()

  fair_thresholder = ThresholdOptimizer(
                  estimator= sk_model,
                  #constraints="equalized_odds",
                  #constraints="false_positive_rate_parity",
                  constraints=constraint,
                  objective="balanced_accuracy_score",
                  flip=True,
                  prefit=True,
                  predict_method='predict')
                
  fair_thresholder.fit(new_train_data_x, new_train_data_y, sensitive_features=new_train_data_sens)

  Y_pred_postprocess = fair_thresholder.predict(new_test_data_x, sensitive_features=new_test_data_sens)

  metrics_dict = {
  'count': count,
  "true_positive_rate": true_positive_rate,
  "false_positive_rate": false_positive_rate,
  'AUROC': roc_auc_score
  }

  mf = MetricFrame(
      metrics=metrics_dict,
      y_true=new_test_data_y,
      y_pred=Y_pred_postprocess,
      sensitive_features=new_test_data_sens
  )
  mf = mf.by_group
  eo_ratio = equalized_odds_ratio(y_true=new_test_data_y, y_pred=Y_pred_postprocess, sensitive_features=new_test_data_sens)

  # Deterministic
  interpolated = fair_thresholder.interpolated_thresholder_
  deterministic_dict = create_deterministic(interpolated.interpolation_dict)

  fair_thresholder_deterministic = InterpolatedThresholder(estimator=interpolated.estimator,
                                                interpolation_dict=deterministic_dict,
                                                prefit=True,
                                                predict_method='predict')

  fair_thresholder_deterministic.fit(new_train_data_x, new_train_data_x, sensitive_features=new_train_data_sens)

  y_pred_postprocess_deterministic = fair_thresholder_deterministic.predict(new_test_data_x, sensitive_features=new_test_data_sens)

  mf_deterministic = MetricFrame(
      metrics=metrics_dict,
      y_true=new_test_data_y,
      y_pred=y_pred_postprocess_deterministic,
      sensitive_features=new_test_data_sens
  )
  mf_deterministic = mf_deterministic.by_group
  eo_ratio_deterministic = equalized_odds_ratio(y_true=new_test_data_y, y_pred=y_pred_postprocess_deterministic, sensitive_features=new_test_data_sens)

  return mf_deterministic, eo_ratio_deterministic, fair_thresholder_deterministic, deterministic_dict, mf, eo_ratio, fair_thresholder, interpolated.interpolation_dict

def create_deterministic(interpolate_dict):
  deterministic_dict = {}
  for (race, operations) in interpolate_dict.items():
    op0, op1 = operations["operation0"]._threshold, operations["operation1"]._threshold
    p0, p1 = operations["p0"], operations["p1"]
    deterministic_dict[race] = Bunch(
      p0=0.0,
      p1=1.0,
      operation0=ThresholdOperation(operator=">",threshold=(p0*op0 + p1*op1)),
      operation1=ThresholdOperation(operator=">",threshold=(p0*op0 + p1*op1))
    )
  return deterministic_dict

def demographics_fun(data):
  # Import
  admissions = pd.read_csv(r"mimic-iv-2.0/hosp/admissions.csv")
  patients = pd.read_csv(r"mimic-iv-2.0/hosp/patients.csv")
  icu_stays = pd.read_csv(r"mimic-iv-2.0/icu/icustays.csv")
  # Filter for relevant columns 
  admissions = admissions[['subject_id', 'hadm_id', 'insurance', 'language', 'marital_status', 'race']]
  patients  = patients[['subject_id', 'gender', 'anchor_age']]
  patients['anchor_age'] = (patients['anchor_age'] / 10).round().astype(int) * 10 # Round age to nearest 10
  icu_stays = icu_stays[['stay_id', 'hadm_id', 'subject_id']]
  # Set type
  admissions['insurance'] = admissions['insurance'].astype("string")
  admissions['language'] = admissions['language'].astype("string")
  admissions['marital_status'] = admissions['marital_status'].astype("string")
  admissions['race'] = admissions['race'].astype("string")
  patients['gender'] = patients['gender'].astype("string")

  # Group race
  pd.options.mode.chained_assignment = None
  admissions['race'] = admissions['race'].str.replace('SOUTH AMERICAN', 'HISPANIC')
  admissions['race'] = admissions['race'].str.replace('MULTIPLE RACE/ETHNICITY', 'OTHER')
  admissions['race'] = admissions['race'].str.replace('PORTUGUESE', 'OTHISPANICHER')
  admissions['race'] = admissions['race'].str.replace('UNABLE TO OBTAIN', 'UNKNOWN')
  admissions['race'] = admissions['race'].str.replace('OTHISPANICHER', 'OTHER')
  admissions['race'] = admissions['race'].str.replace('PATIENT DECLINED TO ANSWER', 'UNKNOWN')

  x = 0
  string_list = ['NATIVE', 'ASIAN', 'HISPANIC', 'BLACK', 'WHITE', 'OTHER', 'UNKNOWN']
  for string in string_list:
      x += 1
      sub_df = admissions[admissions['race'].str.contains(string, case=False, na=False)]
      sub_df['grouped_race'] = string # use filter string as final_label 
      if x == 1:
          new_admissions = sub_df
      else:
          new_admissions = pd.concat([new_admissions, sub_df])
  new_admissions.drop(columns=['race'], inplace=True)

  # Get stays
  stay_list = data.stay_id.unique().tolist()
  # Filter for stays 
  icu_stays = icu_stays[icu_stays['stay_id'].isin(stay_list)]
  # Merge
  demographics = icu_stays.merge(patients)
  demographics = demographics.merge(new_admissions)
  demographics.drop_duplicates(subset=['stay_id', 'hadm_id', 'subject_id', 'gender', 'anchor_age', 'insurance', 'language', 'marital_status'], inplace=True)
  demographics.drop(columns=['subject_id', 'hadm_id'], inplace=True)

  # Fill in nan
  demographics = demographics.fillna('unknown')

  return demographics

def native_fair_performance(model, data, test_data, string='insurance'):

  demographics = demographics_fun(data) 

  # Filter for string
  demographics = demographics[['stay_id', string]]

  # Change column to type string not object ...
  demographics[string] = pd.Series(demographics[string], dtype="string")

  # Get test data
  new_test_data = pd.merge(test_data, demographics)
  new_test_data.sort_values(by=[string], inplace=True)
  new_test_data_y = new_test_data['po_flag']
  new_test_data_sens = new_test_data[[string]]

  Y_pred_postprocess = []

  for x in new_test_data[string].unique():

    temp_data = new_test_data[new_test_data[string] == x].drop(columns=[string])

    # Define batch size 
    batch_size = 256

    temp_dataset = MIMICDataset(temp_data)
    temp_dataloader = DataLoader(dataset=temp_dataset, batch_size=batch_size, collate_fn=temp_dataset.collate_fn_padd)

    # Define loss
    criterion = nn.BCEWithLogitsLoss()

    temp_loss, temp_accuracy, temp_auroc, temp_predictions, temp_labels = evaluate(model, temp_dataloader, criterion)

    Y_pred_postprocess.extend(temp_predictions)
  
  Y_pred_postprocess, Y_pred_postprocess_ub = new_threshold_fun(Y_pred_postprocess)

  metrics_dict = {'count': count,"true_positive_rate": true_positive_rate,"false_positive_rate": false_positive_rate,'AUROC':roc_auc_score}
  
  mf = MetricFrame(metrics=metrics_dict,y_true=new_test_data_y,y_pred=Y_pred_postprocess,sensitive_features=new_test_data_sens).by_group
  
  return mf

# Function to train and eval model 
def fair_run_fun(data_dict, model, string, data, constraint_x="equalized_odds"):

    # Define batch size 
    batch_size = 256

    # Define optimizer and learning_rate
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss
    criterion = nn.BCEWithLogitsLoss()

    # Define epochs and clip
    N_EPOCHS = 10 #10
    CLIP = 1

    origional_insurance_df = pd.DataFrame()
    origional_race_df = pd.DataFrame()
    origional_age_df = pd.DataFrame()
    fair_insurance_df = pd.DataFrame()
    fair_race_df = pd.DataFrame()
    fair_age_df = pd.DataFrame()
    fair_dict_deterministic_insurance_list = []
    fair_dict_deterministic_race_list = []
    fair_dict_deterministic_age_list = []

    # Iterate through dict i.e fold
    for key, value in data_dict.items():
        train_data2 = value[0] # smote
        valid_data = value[1]
        test_data = value[2]
        train_data = value[2] # no smote 

        # Initializing the weights of our model each fold
        model.apply(init_weights)
        
        # Define dataloaders
        train_dataset = MIMICDataset(train_data2)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn_padd)

        valid_dataset = MIMICDataset(valid_data)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn_padd)

        test_dataset = MIMICDataset(test_data)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn_padd)

        # Run
        best_valid_loss = float('inf')
        best_valid_auroc = 0

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_accuracy, train_auroc, train_predictions, train_labels = train(model, train_dataloader, optimizer, criterion, CLIP)
            valid_loss, valid_accuracy, valid_auroc, valid_predictions, valid_labels = evaluate(model, valid_dataloader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #print('BEST VALID LOSS')

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                #print('UPDATED BEST INTERMEDIATE MODEL')
                torch.save(model.state_dict(), f'hold_out_switch_model_intermediate_{string}.pt')

        # -----------------------------
        # Evaluate best model on test set
        # -----------------------------

        model.load_state_dict(torch.load(f'hold_out_switch_model_intermediate_{string}.pt'))

        string_list= ['insurance', 'grouped_race', 'anchor_age']

        for fair_string in string_list:

            origional_mf = native_fair_performance(model, data, test_data, fair_string)

            mf_deterministic, eo_ratio_deterministic, fair_est_deterministic, fair_dict_deterministic, mf, eo_ratio, fair_est, fair_dict = fairness_fun(data, train_data, test_data, fair_string, constraint=constraint_x, cv=True)

            #print(origional_mf)
            #print(mf_deterministic)

            if fair_string == 'insurance':
                origional_insurance_df = pd.concat((origional_insurance_df, origional_mf))
                fair_insurance_df = pd.concat((fair_insurance_df, mf_deterministic))
                fair_dict_deterministic_insurance_list.append(fair_dict_deterministic)
            elif fair_string == 'grouped_race':
                origional_race_df = pd.concat((origional_race_df, origional_mf))
                fair_race_df = pd.concat((fair_race_df, mf_deterministic))
                fair_dict_deterministic_race_list.append(fair_dict_deterministic)
            elif fair_string == 'anchor_age':
                origional_age_df = pd.concat((origional_age_df, origional_mf))
                fair_age_df = pd.concat((fair_age_df, mf_deterministic))
                fair_dict_deterministic_age_list.append(fair_dict_deterministic)

    # Get mean
    by_row_index = origional_insurance_df.groupby(origional_insurance_df.index)
    origional_insurance_df = by_row_index.mean()
    by_row_index = fair_insurance_df.groupby(fair_insurance_df.index)
    fair_insurance_df = by_row_index.mean()
    by_row_index = origional_race_df.groupby(origional_race_df.index)
    origional_race_df = by_row_index.mean()
    by_row_index = fair_race_df.groupby(fair_race_df.index)
    fair_race_df = by_row_index.mean()
    by_row_index = origional_age_df.groupby(origional_age_df.index)
    origional_age_df = by_row_index.mean()
    by_row_index = fair_age_df.groupby(fair_age_df.index)
    fair_age_df = by_row_index.mean()

    return origional_insurance_df, fair_insurance_df, fair_dict_deterministic_insurance_list, origional_race_df, fair_race_df, fair_dict_deterministic_race_list, origional_age_df, fair_age_df, fair_dict_deterministic_age_list

# Function to split data so even distribution between val and test
def fair_data_fun(data, individual, n_cv=10):

    demographics = demographics_fun(data)
    
    data_dict = {}
    random_x_list = []
    x = -1

    for i in range(n_cv):
        x += 1
        stays = data['stay_id'].unique()
        random.Random(x).shuffle(stays) 
        X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
        # Filter for features in this individual
        X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
        model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
        model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
        n = round(0.7 * len(stays))
        n2 = round(0.85 * len(stays))
        train_stays = stays[:n]
        validation_stays = stays[n:n2]
        test_stays = stays[n2:]
        train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
        valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
        test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
        train_data2 = smote_fun(train_data)

        # Filter for fair labels
        while not (pd.merge(test_data, demographics).groupby('insurance').po_flag.nunique() > 1).all() & (pd.merge(test_data, demographics).groupby('grouped_race').po_flag.nunique() > 1).all() & (pd.merge(test_data, demographics).groupby('anchor_age').po_flag.nunique() > 1).all():
            x +=1
            #if x == 25 or x == 33:
            #    x += 1
            #print('x', x)
            stays = data['stay_id'].unique()
            random.Random(x).shuffle(stays) 
            X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
            # Filter for features in this individual
            X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
            model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
            model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
            n = round(0.7 * len(stays))
            n2 = round(0.85 * len(stays))
            train_stays = stays[:n]
            validation_stays = stays[n:n2]
            test_stays = stays[n2:]
            train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
            valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
            test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
            train_data2 = smote_fun(train_data)
 
        while not math.isclose(test_data.po_flag.value_counts(normalize=True)[1], valid_data.po_flag.value_counts(normalize=True)[1], abs_tol=0.005): # Check to make sure val and test set are comparable
            x +=1
            #if x == 25 or x == 33:
            #    x += 1
            #print('x', x)
            stays = data['stay_id'].unique()
            random.Random(x).shuffle(stays) 
            X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
            # Filter for features in this individual
            X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
            model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
            model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
            n = round(0.7 * len(stays))
            n2 = round(0.85 * len(stays))
            train_stays = stays[:n]
            validation_stays = stays[n:n2]
            test_stays = stays[n2:]
            train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
            valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
            test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
            train_data2 = smote_fun(train_data)

            # Filter for fair labels
            while not (pd.merge(test_data, demographics).groupby('insurance').po_flag.nunique() > 1).all() & (pd.merge(test_data, demographics).groupby('grouped_race').po_flag.nunique() > 1).all() & (pd.merge(test_data, demographics).groupby('anchor_age').po_flag.nunique() > 1).all():
                x +=1
                #if x == 25 or x == 33:
                #    x += 1
                #print('x', x)
                stays = data['stay_id'].unique()
                random.Random(x).shuffle(stays)
                X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
                # Filter for features in this individual
                X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
                model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
                model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
                n = round(0.7 * len(stays))
                n2 = round(0.85 * len(stays))
                train_stays = stays[:n]
                validation_stays = stays[n:n2]
                test_stays = stays[n2:]
                train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
                valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
                test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
                train_data2 = smote_fun(train_data)

        data_dict[i] = [train_data2, valid_data, test_data, train_data]
        random_x_list.append(x)

    return data_dict, random_x_list

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    axe.set_ylabel('Count')

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    
    return axe

def plot_clustered_stacked_2(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    axe.set_ylabel('Percentage')
    axe.set_ylim(0, 100)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    
    return axe

# Function to train and eval model 
def bad_bio_run_fun(data_dict, model, string, bad_bio_stay_id_list):

    #overall_best_valid_auroc = 0
    overall_best_test_auroc = 0

    test_auroc_results = []
    test_accuracy_results = []
    test_balanced_accuracy_results = []
    test_recall_results = []
    test_precision_results = []
    test_f1_results = []
    test_auprc_results = []
    test_cm_results = []
    test_true_positive_rate_results = []
    test_fasle_positive_rate_results = []

    ub_test_auroc_results = []
    ub_test_accuracy_results = []
    ub_test_balanced_accuracy_results = []
    ub_test_recall_results = []
    ub_test_precision_results = []
    ub_test_f1_results = []
    ub_test_auprc_results = []
    ub_test_cm_results = []
    ub_test_true_positive_rate_results = []
    ub_test_fasle_positive_rate_results = []

    master_equalised_odds_df = pd.DataFrame()

    # Define batch size 
    batch_size = 256

    # Define optimizer and learning_rate
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss
    criterion = nn.BCEWithLogitsLoss()

    # Define epochs and clip
    N_EPOCHS = 10 #10
    CLIP = 1

    # Iterate through dict i.e fold
    for key, value in data_dict.items():
        train_data = value[0]
        valid_data = value[1]
        test_data = value[2]

        # Filter for bad bio
        test_data = test_data[test_data['stay_id'].isin(bad_bio_stay_id_list)]

        # Initializing the weights of our model each fold
        model.apply(init_weights)
        
        # Define dataloaders
        train_dataset = MIMICDataset(train_data)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn_padd)

        valid_dataset = MIMICDataset(valid_data)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn_padd)

        test_dataset = MIMICDataset(test_data)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn_padd)

        # Run
        best_valid_loss = float('inf')
        best_valid_auroc = 0

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_accuracy, train_auroc, train_predictions, train_labels = train(model, train_dataloader, optimizer, criterion, CLIP)
            valid_loss, valid_accuracy, valid_auroc, valid_predictions, valid_labels = evaluate(model, valid_dataloader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #print('BEST VALID LOSS')

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                #print('UPDATED BEST INTERMEDIATE MODEL')
                torch.save(model.state_dict(), f'hold_out_switch_model_intermediate_{string}.pt')

        # -----------------------------
        # Evaluate best model on test set
        # -----------------------------

        model.load_state_dict(torch.load(f'hold_out_switch_model_intermediate_{string}.pt'))

        test_loss, test_accuracy, test_auroc, test_predictions, test_labels = evaluate(model, test_dataloader, criterion)

        print('Test AUROC result:', test_auroc)

        # Use new cut off
        lower_bound_test_predictions, upper_bound_test_predictions = new_threshold_fun(test_predictions)

        # Lower bound
        test_auroc2 = roc_auc_score(test_labels, lower_bound_test_predictions)
        print('Test AUROC result 2:', test_auroc2)
        test_accuracy2 = accuracy_score(test_labels, lower_bound_test_predictions)
        #assert test_accuracy == test_accuracy2
        test_balanced_accuracy = balanced_accuracy_score(test_labels, lower_bound_test_predictions)
        test_recall = recall_score(test_labels, lower_bound_test_predictions)
        test_precision = precision_score(test_labels, lower_bound_test_predictions)
        test_f1 = f1_score(test_labels, lower_bound_test_predictions)
        test_auprc = average_precision_score(test_labels, lower_bound_test_predictions)
        test_cm = confusion_matrix(test_labels, lower_bound_test_predictions)
        tn, fp, fn, tp = test_cm.ravel()
        test_true_positive_rate = (tp / (tp + fn))
        test_false_positive_rate = (fp / (fp + tn))

        # Upper bound
        ub_test_auroc2 = roc_auc_score(test_labels, upper_bound_test_predictions)
        ub_test_accuracy2 = accuracy_score(test_labels, upper_bound_test_predictions)
        ub_test_balanced_accuracy = balanced_accuracy_score(test_labels, upper_bound_test_predictions)
        ub_test_recall = recall_score(test_labels, upper_bound_test_predictions)
        ub_test_precision = precision_score(test_labels, upper_bound_test_predictions)
        ub_test_f1 = f1_score(test_labels, upper_bound_test_predictions)
        ub_test_auprc = average_precision_score(test_labels, upper_bound_test_predictions)
        ub_test_cm = confusion_matrix(test_labels, upper_bound_test_predictions)
        tn, fp, fn, tp = ub_test_cm.ravel()
        ub_test_true_positive_rate = (tp / (tp + fn))
        ub_test_false_positive_rate = (fp / (fp + tn))

        # Check fairness
        equalised_odds_df = equalised_odds(test_data, batch_size, model, criterion)
        master_equalised_odds_df = pd.concat([master_equalised_odds_df, equalised_odds_df], axis=0)
        #print(master_equalised_odds_df)
        
        #confusion_matrix(test_labels.round(), test_predictions.round())

        #if test_auroc2 > overall_best_test_auroc:
            #overall_best_test_auroc = test_auroc2
            #print('UPDATED BEST OVERALL MODEL')
            #torch.save(model.state_dict(), f'hold_out_switch_model_{string}.pt')
        
        test_auroc_results.append(test_auroc2)
        test_accuracy_results.append(test_accuracy2)
        test_balanced_accuracy_results.append(test_balanced_accuracy)
        test_recall_results.append(test_recall)
        test_precision_results.append(test_precision)
        test_f1_results.append(test_f1)
        test_auprc_results.append(test_auprc)
        test_cm_results.append(test_cm)
        test_true_positive_rate_results.append(test_true_positive_rate)
        test_fasle_positive_rate_results.append(test_false_positive_rate)

        ub_test_auroc_results.append(ub_test_auroc2)
        ub_test_accuracy_results.append(ub_test_accuracy2)
        ub_test_balanced_accuracy_results.append(ub_test_balanced_accuracy)
        ub_test_recall_results.append(ub_test_recall)
        ub_test_precision_results.append(ub_test_precision)
        ub_test_f1_results.append(ub_test_f1)
        ub_test_auprc_results.append(ub_test_auprc)
        ub_test_cm_results.append(ub_test_cm)
        ub_test_true_positive_rate_results.append(ub_test_true_positive_rate)
        ub_test_fasle_positive_rate_results.append(ub_test_false_positive_rate)

    test_results = [test_auroc_results, test_accuracy_results,
        test_balanced_accuracy_results,
        test_recall_results,
        test_precision_results,
        test_f1_results,
        test_auprc_results,
        test_cm_results,
        test_true_positive_rate_results,
        test_fasle_positive_rate_results
        ]
    
    ub_test_results = [ub_test_auroc_results, ub_test_accuracy_results,
        ub_test_balanced_accuracy_results,
        ub_test_recall_results,
        ub_test_precision_results,
        ub_test_f1_results,
        ub_test_auprc_results,
        ub_test_cm_results,
        ub_test_true_positive_rate_results,
        ub_test_fasle_positive_rate_results
        ]
       
    master_equalised_odds_df.set_index(['column', 'value'], inplace=True)
    by_row_index = master_equalised_odds_df.groupby(master_equalised_odds_df.index)
    mean_equalised_odds_df = by_row_index.mean()
    sd_equalised_odds_df = by_row_index.std()

    return test_results, ub_test_results, mean_equalised_odds_df, sd_equalised_odds_df

# Function to train and eval model 
def eicu_run_fun(data_dict, model, string):

    #overall_best_valid_auroc = 0
    overall_best_test_auroc = 0

    test_auroc_results = []
    test_accuracy_results = []
    test_balanced_accuracy_results = []
    test_recall_results = []
    test_precision_results = []
    test_f1_results = []
    test_auprc_results = []
    test_cm_results = []
    test_true_positive_rate_results = []
    test_fasle_positive_rate_results = []

    ub_test_auroc_results = []
    ub_test_accuracy_results = []
    ub_test_balanced_accuracy_results = []
    ub_test_recall_results = []
    ub_test_precision_results = []
    ub_test_f1_results = []
    ub_test_auprc_results = []
    ub_test_cm_results = []
    ub_test_true_positive_rate_results = []
    ub_test_fasle_positive_rate_results = []

    master_equalised_odds_df = pd.DataFrame()

    # Define batch size 
    batch_size = 256

    # Define optimizer and learning_rate
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss
    criterion = nn.BCEWithLogitsLoss()

    # Define epochs and clip
    N_EPOCHS = 10 #10
    CLIP = 1

    # Iterate through dict i.e fold
    for key, value in data_dict.items():
        train_data = value[0]
        valid_data = value[1]
        test_data = value[2]

        # Load best mimic model each fold
        #model.apply(init_weights)
        model.load_state_dict(torch.load('hold_out_switch_model_short.pt'))
        
        # Define dataloaders
        train_dataset = MIMICDataset(train_data)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn_padd)

        valid_dataset = MIMICDataset(valid_data)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn_padd)

        test_dataset = MIMICDataset(test_data)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn_padd)

        # Run
        best_valid_loss = float('inf')
        best_valid_auroc = 0

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_accuracy, train_auroc, train_predictions, train_labels = train(model, train_dataloader, optimizer, criterion, CLIP)
            valid_loss, valid_accuracy, valid_auroc, valid_predictions, valid_labels = evaluate(model, valid_dataloader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #print('BEST VALID LOSS')

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                #print('UPDATED BEST INTERMEDIATE MODEL')
                torch.save(model.state_dict(), f'eicu_switch_model_intermediate_{string}.pt')

        # -----------------------------
        # Evaluate best model on test set
        # -----------------------------

        model.load_state_dict(torch.load(f'eicu_switch_model_intermediate_{string}.pt'))

        test_loss, test_accuracy, test_auroc, test_predictions, test_labels = evaluate(model, test_dataloader, criterion)

        print('Test AUROC result:', test_auroc)

        # Use new cut off
        lower_bound_test_predictions, upper_bound_test_predictions = new_threshold_fun(test_predictions)

        # Lower bound
        test_auroc2 = roc_auc_score(test_labels, lower_bound_test_predictions)
        print('Test AUROC result 2:', test_auroc2)
        test_accuracy2 = accuracy_score(test_labels, lower_bound_test_predictions)
        #assert test_accuracy == test_accuracy2
        test_balanced_accuracy = balanced_accuracy_score(test_labels, lower_bound_test_predictions)
        test_recall = recall_score(test_labels, lower_bound_test_predictions)
        test_precision = precision_score(test_labels, lower_bound_test_predictions)
        test_f1 = f1_score(test_labels, lower_bound_test_predictions)
        test_auprc = average_precision_score(test_labels, lower_bound_test_predictions)
        test_cm = confusion_matrix(test_labels, lower_bound_test_predictions)
        tn, fp, fn, tp = test_cm.ravel()
        test_true_positive_rate = (tp / (tp + fn))
        test_false_positive_rate = (fp / (fp + tn))

        # Upper bound
        ub_test_auroc2 = roc_auc_score(test_labels, upper_bound_test_predictions)
        ub_test_accuracy2 = accuracy_score(test_labels, upper_bound_test_predictions)
        ub_test_balanced_accuracy = balanced_accuracy_score(test_labels, upper_bound_test_predictions)
        ub_test_recall = recall_score(test_labels, upper_bound_test_predictions)
        ub_test_precision = precision_score(test_labels, upper_bound_test_predictions)
        ub_test_f1 = f1_score(test_labels, upper_bound_test_predictions)
        ub_test_auprc = average_precision_score(test_labels, upper_bound_test_predictions)
        ub_test_cm = confusion_matrix(test_labels, upper_bound_test_predictions)
        tn, fp, fn, tp = ub_test_cm.ravel()
        ub_test_true_positive_rate = (tp / (tp + fn))
        ub_test_false_positive_rate = (fp / (fp + tn))
        
        if test_auroc2 > overall_best_test_auroc:
            overall_best_test_auroc = test_auroc2
            print('UPDATED BEST OVERALL MODEL')
            torch.save(model.state_dict(), f'eicu_switch_model_{string}.pt')
        
        test_auroc_results.append(test_auroc2)
        test_accuracy_results.append(test_accuracy2)
        test_balanced_accuracy_results.append(test_balanced_accuracy)
        test_recall_results.append(test_recall)
        test_precision_results.append(test_precision)
        test_f1_results.append(test_f1)
        test_auprc_results.append(test_auprc)
        test_cm_results.append(test_cm)
        test_true_positive_rate_results.append(test_true_positive_rate)
        test_fasle_positive_rate_results.append(test_false_positive_rate)

        ub_test_auroc_results.append(ub_test_auroc2)
        ub_test_accuracy_results.append(ub_test_accuracy2)
        ub_test_balanced_accuracy_results.append(ub_test_balanced_accuracy)
        ub_test_recall_results.append(ub_test_recall)
        ub_test_precision_results.append(ub_test_precision)
        ub_test_f1_results.append(ub_test_f1)
        ub_test_auprc_results.append(ub_test_auprc)
        ub_test_cm_results.append(ub_test_cm)
        ub_test_true_positive_rate_results.append(ub_test_true_positive_rate)
        ub_test_fasle_positive_rate_results.append(ub_test_false_positive_rate)

    test_results = [test_auroc_results, test_accuracy_results,
        test_balanced_accuracy_results,
        test_recall_results,
        test_precision_results,
        test_f1_results,
        test_auprc_results,
        test_cm_results,
        test_true_positive_rate_results,
        test_fasle_positive_rate_results
        ]
    
    ub_test_results = [ub_test_auroc_results, ub_test_accuracy_results,
        ub_test_balanced_accuracy_results,
        ub_test_recall_results,
        ub_test_precision_results,
        ub_test_f1_results,
        ub_test_auprc_results,
        ub_test_cm_results,
        ub_test_true_positive_rate_results,
        ub_test_fasle_positive_rate_results
        ]

    return test_results, ub_test_results

# Function to train and eval model 
def bad_abs_run_fun(data_dict, model, string, tpn_stay_list):

    #overall_best_valid_auroc = 0
    overall_best_test_auroc = 0

    test_auroc_results = []
    test_accuracy_results = []
    test_balanced_accuracy_results = []
    test_recall_results = []
    test_precision_results = []
    test_f1_results = []
    test_auprc_results = []
    test_cm_results = []
    test_true_positive_rate_results = []
    test_fasle_positive_rate_results = []

    ub_test_auroc_results = []
    ub_test_accuracy_results = []
    ub_test_balanced_accuracy_results = []
    ub_test_recall_results = []
    ub_test_precision_results = []
    ub_test_f1_results = []
    ub_test_auprc_results = []
    ub_test_cm_results = []
    ub_test_true_positive_rate_results = []
    ub_test_fasle_positive_rate_results = []

    master_equalised_odds_df = pd.DataFrame()

    # Define batch size 
    batch_size = 256

    # Define optimizer and learning_rate
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss
    criterion = nn.BCEWithLogitsLoss()

    # Define epochs and clip
    N_EPOCHS = 10 #10
    CLIP = 1

    # Iterate through dict i.e fold
    for key, value in data_dict.items():
        train_data = value[0]
        valid_data = value[1]
        test_data = value[2]

        # Filter for bad bio
        test_data = test_data[test_data['stay_id'].isin(tpn_stay_list)]

        # Initializing the weights of our model each fold
        model.apply(init_weights)
        
        # Define dataloaders
        train_dataset = MIMICDataset(train_data)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn_padd)

        valid_dataset = MIMICDataset(valid_data)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn_padd)

        test_dataset = MIMICDataset(test_data)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn_padd)

        # Run
        best_valid_loss = float('inf')
        best_valid_auroc = 0

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_accuracy, train_auroc, train_predictions, train_labels = train(model, train_dataloader, optimizer, criterion, CLIP)
            valid_loss, valid_accuracy, valid_auroc, valid_predictions, valid_labels = evaluate(model, valid_dataloader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #print('BEST VALID LOSS')

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                #print('UPDATED BEST INTERMEDIATE MODEL')
                torch.save(model.state_dict(), f'hold_out_switch_model_intermediate_{string}.pt')

        # -----------------------------
        # Evaluate best model on test set
        # -----------------------------

        model.load_state_dict(torch.load(f'hold_out_switch_model_intermediate_{string}.pt'))

        test_loss, test_accuracy, test_auroc, test_predictions, test_labels = evaluate(model, test_dataloader, criterion)

        print('Test AUROC result:', test_auroc)

        # Use new cut off
        lower_bound_test_predictions, upper_bound_test_predictions = new_threshold_fun(test_predictions)

        # Lower bound
        try:
            test_auroc2 = roc_auc_score(test_labels, lower_bound_test_predictions)
        except:
            test_auroc2 = np.nan
        print('Test AUROC result 2:', test_auroc2)
        test_accuracy2 = accuracy_score(test_labels, lower_bound_test_predictions)
        #assert test_accuracy == test_accuracy2
        try:
            test_balanced_accuracy = balanced_accuracy_score(test_labels, lower_bound_test_predictions)
        except:
            test_balanced_accuracy = np.nan
        test_recall = recall_score(test_labels, lower_bound_test_predictions)
        test_precision = precision_score(test_labels, lower_bound_test_predictions)
        test_f1 = f1_score(test_labels, lower_bound_test_predictions)
        test_auprc = average_precision_score(test_labels, lower_bound_test_predictions)
        test_cm = confusion_matrix(test_labels, lower_bound_test_predictions)
        if test_cm.shape == (2, 2):
            tn, fp, fn, tp = test_cm.ravel()
            test_true_positive_rate = (tp / (tp + fn))
            test_false_positive_rate = (fp / (fp + tn))
        else:
            test_true_positive_rate = np.nan
            test_false_positive_rate = np.nan

        # Upper bound
        try:
            ub_test_auroc2 = roc_auc_score(test_labels, upper_bound_test_predictions)
        except:
            ub_test_auroc2 = np.nan
        ub_test_accuracy2 = accuracy_score(test_labels, upper_bound_test_predictions)
        try:
            ub_test_balanced_accuracy = balanced_accuracy_score(test_labels, upper_bound_test_predictions)
        except:
            ub_test_balanced_accuracy = np.nan
        ub_test_recall = recall_score(test_labels, upper_bound_test_predictions)
        ub_test_precision = precision_score(test_labels, upper_bound_test_predictions)
        ub_test_f1 = f1_score(test_labels, upper_bound_test_predictions)
        ub_test_auprc = average_precision_score(test_labels, upper_bound_test_predictions)
        ub_test_cm = confusion_matrix(test_labels, upper_bound_test_predictions)
        if test_cm.shape == (2, 2):
            tn, fp, fn, tp = ub_test_cm.ravel()
            ub_test_true_positive_rate = (tp / (tp + fn))
            ub_test_false_positive_rate = (fp / (fp + tn))
        else:
            ub_test_true_positive_rate = np.nan
            ub_test_false_positive_rate = np.nan

        # Check fairness
        equalised_odds_df = equalised_odds(test_data, batch_size, model, criterion)
        master_equalised_odds_df = pd.concat([master_equalised_odds_df, equalised_odds_df], axis=0)
        
        test_auroc_results.append(test_auroc2)
        test_accuracy_results.append(test_accuracy2)
        test_balanced_accuracy_results.append(test_balanced_accuracy)
        test_recall_results.append(test_recall)
        test_precision_results.append(test_precision)
        test_f1_results.append(test_f1)
        test_auprc_results.append(test_auprc)
        test_cm_results.append(test_cm)
        test_true_positive_rate_results.append(test_true_positive_rate)
        test_fasle_positive_rate_results.append(test_false_positive_rate)

        ub_test_auroc_results.append(ub_test_auroc2)
        ub_test_accuracy_results.append(ub_test_accuracy2)
        ub_test_balanced_accuracy_results.append(ub_test_balanced_accuracy)
        ub_test_recall_results.append(ub_test_recall)
        ub_test_precision_results.append(ub_test_precision)
        ub_test_f1_results.append(ub_test_f1)
        ub_test_auprc_results.append(ub_test_auprc)
        ub_test_cm_results.append(ub_test_cm)
        ub_test_true_positive_rate_results.append(ub_test_true_positive_rate)
        ub_test_fasle_positive_rate_results.append(ub_test_false_positive_rate)

    test_results = [test_auroc_results, test_accuracy_results,
        test_balanced_accuracy_results,
        test_recall_results,
        test_precision_results,
        test_f1_results,
        test_auprc_results,
        test_cm_results,
        test_true_positive_rate_results,
        test_fasle_positive_rate_results
        ]
    
    ub_test_results = [ub_test_auroc_results, ub_test_accuracy_results,
        ub_test_balanced_accuracy_results,
        ub_test_recall_results,
        ub_test_precision_results,
        ub_test_f1_results,
        ub_test_auprc_results,
        ub_test_cm_results,
        ub_test_true_positive_rate_results,
        ub_test_fasle_positive_rate_results
        ]
    
    master_equalised_odds_df.set_index(['column', 'value'], inplace=True)
    by_row_index = master_equalised_odds_df.groupby(master_equalised_odds_df.index)
    mean_equalised_odds_df = by_row_index.mean()
    sd_equalised_odds_df = by_row_index.std()

    return test_results, ub_test_results, mean_equalised_odds_df, sd_equalised_odds_df

# Function to split data so even distribution between val and test
def bad_abs_data_fun(tpn_stay_list, data, individual, n_cv=10):
    
    data_dict = {}
    random_x_list = []
    x = -1

    for i in range(n_cv):
        x += 1
        if x == 25 or x == 33:
            x += 1
        #print('x', x)
        stays = data['stay_id'].unique()
        random.Random(x).shuffle(stays)
        X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
        # Filter for features in this individual
        X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
        model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
        model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
        n = round(0.7 * len(stays))
        n2 = round(0.85 * len(stays))
        train_stays = stays[:n]
        validation_stays = stays[n:n2]
        test_stays = stays[n2:]
        train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
        valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
        test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
        # Oversample train set
        train_data = smote_fun(train_data)

        # Filter for bad bio
        while len(test_data[test_data['stay_id'].isin(tpn_stay_list)].po_flag.value_counts(normalize=True)) < 2:
            x +=1
            if x == 25 or x == 33:
                x += 1
            #print('x', x)
            stays = data['stay_id'].unique()
            random.Random(x).shuffle(stays) 
            X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
            # Filter for features in this individual
            X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
            model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
            model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
            n = round(0.7 * len(stays))
            n2 = round(0.85 * len(stays))
            train_stays = stays[:n]
            validation_stays = stays[n:n2]
            test_stays = stays[n2:]
            train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
            valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
            test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
            # Oversample train set
            train_data = smote_fun(train_data)
 
        while not math.isclose(test_data.po_flag.value_counts(normalize=True)[1], valid_data.po_flag.value_counts(normalize=True)[1], abs_tol=0.005): # Check to make sure val and test set are comparable
            x +=1
            if x == 25 or x == 33:
                x += 1
            #print('x', x)
            stays = data['stay_id'].unique()
            random.Random(x).shuffle(stays)
            X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
            # Filter for features in this individual
            X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
            model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
            model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
            n = round(0.7 * len(stays))
            n2 = round(0.85 * len(stays))
            train_stays = stays[:n]
            validation_stays = stays[n:n2]
            test_stays = stays[n2:]
            train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
            valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
            test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
            # Oversample train set
            train_data = smote_fun(train_data)

            # Filter for bad bio
            while len(test_data[test_data['stay_id'].isin(tpn_stay_list)].po_flag.value_counts(normalize=True)) < 2:
                x +=1
                if x == 25 or x == 33:
                    x += 1
                #print('x', x)
                stays = data['stay_id'].unique()
                random.Random(x).shuffle(stays)
                X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
                # Filter for features in this individual
                X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
                model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
                model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
                n = round(0.7 * len(stays))
                n2 = round(0.85 * len(stays))
                train_stays = stays[:n]
                validation_stays = stays[n:n2]
                test_stays = stays[n2:]
                train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
                valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
                test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
                # Oversample train set
                train_data = smote_fun(train_data)

        data_dict[i] = [train_data, valid_data, test_data]
        random_x_list.append(x)

    return data_dict, random_x_list


# Function to train and eval model 
def diagnosis_run_fun(data_dict, model, string, filter_list):

    #overall_best_valid_auroc = 0
    overall_best_test_auroc = 0

    test_auroc_results = []
    test_accuracy_results = []
    test_balanced_accuracy_results = []
    test_recall_results = []
    test_precision_results = []
    test_f1_results = []
    test_auprc_results = []
    test_cm_results = []
    test_true_positive_rate_results = []
    test_fasle_positive_rate_results = []

    ub_test_auroc_results = []
    ub_test_accuracy_results = []
    ub_test_balanced_accuracy_results = []
    ub_test_recall_results = []
    ub_test_precision_results = []
    ub_test_f1_results = []
    ub_test_auprc_results = []
    ub_test_cm_results = []
    ub_test_true_positive_rate_results = []
    ub_test_fasle_positive_rate_results = []

    master_equalised_odds_df = pd.DataFrame()

    # Define batch size 
    batch_size = 256

    # Define optimizer and learning_rate
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss
    criterion = nn.BCEWithLogitsLoss()

    # Define epochs and clip
    N_EPOCHS = 10 #10
    CLIP = 1

    # Iterate through dict i.e fold
    for key, value in data_dict.items():
        train_data = value[0]
        valid_data = value[1]
        test_data = value[2]

        # Filter for bad bio
        test_data = test_data[test_data['stay_id'].isin(filter_list)]

        # Initializing the weights of our model each fold
        model.apply(init_weights)
        
        # Define dataloaders
        train_dataset = MIMICDataset(train_data)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn_padd)

        valid_dataset = MIMICDataset(valid_data)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn_padd)

        test_dataset = MIMICDataset(test_data)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn_padd)

        # Run
        best_valid_loss = float('inf')
        best_valid_auroc = 0

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_accuracy, train_auroc, train_predictions, train_labels = train(model, train_dataloader, optimizer, criterion, CLIP)
            valid_loss, valid_accuracy, valid_auroc, valid_predictions, valid_labels = evaluate(model, valid_dataloader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #print('BEST VALID LOSS')

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                #print('UPDATED BEST INTERMEDIATE MODEL')
                torch.save(model.state_dict(), f'hold_out_switch_model_intermediate_{string}.pt')

        # -----------------------------
        # Evaluate best model on test set
        # -----------------------------

        model.load_state_dict(torch.load(f'hold_out_switch_model_intermediate_{string}.pt'))

        test_loss, test_accuracy, test_auroc, test_predictions, test_labels = evaluate(model, test_dataloader, criterion)

        print('Test AUROC result:', test_auroc)

        # Use new cut off
        lower_bound_test_predictions, upper_bound_test_predictions = new_threshold_fun(test_predictions)

        # Lower bound
        try:
            test_auroc2 = roc_auc_score(test_labels, lower_bound_test_predictions)
        except:
            test_auroc2 = np.nan
        print('Test AUROC result 2:', test_auroc2)
        test_accuracy2 = accuracy_score(test_labels, lower_bound_test_predictions)
        #assert test_accuracy == test_accuracy2
        try:
            test_balanced_accuracy = balanced_accuracy_score(test_labels, lower_bound_test_predictions)
        except:
            test_balanced_accuracy = np.nan
        test_recall = recall_score(test_labels, lower_bound_test_predictions)
        test_precision = precision_score(test_labels, lower_bound_test_predictions)
        test_f1 = f1_score(test_labels, lower_bound_test_predictions)
        test_auprc = average_precision_score(test_labels, lower_bound_test_predictions)
        test_cm = confusion_matrix(test_labels, lower_bound_test_predictions)
        if test_cm.shape == (2, 2):
            tn, fp, fn, tp = test_cm.ravel()
            test_true_positive_rate = (tp / (tp + fn))
            test_false_positive_rate = (fp / (fp + tn))
        else:
            test_true_positive_rate = np.nan
            test_false_positive_rate = np.nan

        # Upper bound
        try:
            ub_test_auroc2 = roc_auc_score(test_labels, upper_bound_test_predictions)
        except:
            ub_test_auroc2 = np.nan
        ub_test_accuracy2 = accuracy_score(test_labels, upper_bound_test_predictions)
        try:
            ub_test_balanced_accuracy = balanced_accuracy_score(test_labels, upper_bound_test_predictions)
        except:
            ub_test_balanced_accuracy = np.nan
        ub_test_recall = recall_score(test_labels, upper_bound_test_predictions)
        ub_test_precision = precision_score(test_labels, upper_bound_test_predictions)
        ub_test_f1 = f1_score(test_labels, upper_bound_test_predictions)
        ub_test_auprc = average_precision_score(test_labels, upper_bound_test_predictions)
        ub_test_cm = confusion_matrix(test_labels, upper_bound_test_predictions)
        if test_cm.shape == (2, 2):
            tn, fp, fn, tp = ub_test_cm.ravel()
            ub_test_true_positive_rate = (tp / (tp + fn))
            ub_test_false_positive_rate = (fp / (fp + tn))
        else:
            ub_test_true_positive_rate = np.nan
            ub_test_false_positive_rate = np.nan

        # Check fairness
        equalised_odds_df = equalised_odds(test_data, batch_size, model, criterion)
        master_equalised_odds_df = pd.concat([master_equalised_odds_df, equalised_odds_df], axis=0)        
        
        test_auroc_results.append(test_auroc2)
        test_accuracy_results.append(test_accuracy2)
        test_balanced_accuracy_results.append(test_balanced_accuracy)
        test_recall_results.append(test_recall)
        test_precision_results.append(test_precision)
        test_f1_results.append(test_f1)
        test_auprc_results.append(test_auprc)
        test_cm_results.append(test_cm)
        test_true_positive_rate_results.append(test_true_positive_rate)
        test_fasle_positive_rate_results.append(test_false_positive_rate)

        ub_test_auroc_results.append(ub_test_auroc2)
        ub_test_accuracy_results.append(ub_test_accuracy2)
        ub_test_balanced_accuracy_results.append(ub_test_balanced_accuracy)
        ub_test_recall_results.append(ub_test_recall)
        ub_test_precision_results.append(ub_test_precision)
        ub_test_f1_results.append(ub_test_f1)
        ub_test_auprc_results.append(ub_test_auprc)
        ub_test_cm_results.append(ub_test_cm)
        ub_test_true_positive_rate_results.append(ub_test_true_positive_rate)
        ub_test_fasle_positive_rate_results.append(ub_test_false_positive_rate)

    test_results = [test_auroc_results, test_accuracy_results,
        test_balanced_accuracy_results,
        test_recall_results,
        test_precision_results,
        test_f1_results,
        test_auprc_results,
        test_cm_results,
        test_true_positive_rate_results,
        test_fasle_positive_rate_results
        ]
    
    ub_test_results = [ub_test_auroc_results, ub_test_accuracy_results,
        ub_test_balanced_accuracy_results,
        ub_test_recall_results,
        ub_test_precision_results,
        ub_test_f1_results,
        ub_test_auprc_results,
        ub_test_cm_results,
        ub_test_true_positive_rate_results,
        ub_test_fasle_positive_rate_results
        ]
    
    master_equalised_odds_df.set_index(['column', 'value'], inplace=True)
    by_row_index = master_equalised_odds_df.groupby(master_equalised_odds_df.index)
    mean_equalised_odds_df = by_row_index.mean()
    sd_equalised_odds_df = by_row_index.std()

    return test_results, ub_test_results, mean_equalised_odds_df, sd_equalised_odds_df

# Function to split data so even distribution between val and test
def sepsis_data_fun(sepsis_stays, data, individual, n_cv=10):
    
    data_dict = {}
    random_x_list = []
    x = -1

    for i in range(n_cv):
        x += 1
        #if x == 25 or x == 33:
        #    x += 1
        #print('x', x)
        stays = data['stay_id'].unique()
        random.Random(x).shuffle(stays)
        X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
        # Filter for features in this individual
        X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
        model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
        model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
        n = round(0.7 * len(stays))
        n2 = round(0.85 * len(stays))
        train_stays = stays[:n]
        validation_stays = stays[n:n2]
        test_stays = stays[n2:]
        train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
        valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
        test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
        # Oversample train set
        train_data = smote_fun(train_data)

        # Filter for bad bio
        while len(test_data[test_data['stay_id'].isin(sepsis_stays)].po_flag.value_counts(normalize=True)) < 2:
            x +=1
            #if x == 25 or x == 33:
            #    x += 1
            #print('x', x)
            stays = data['stay_id'].unique()
            random.Random(x).shuffle(stays) 
            X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
            # Filter for features in this individual
            X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
            model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
            model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
            n = round(0.7 * len(stays))
            n2 = round(0.85 * len(stays))
            train_stays = stays[:n]
            validation_stays = stays[n:n2]
            test_stays = stays[n2:]
            train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
            valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
            test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
            # Oversample train set
            train_data = smote_fun(train_data)
 
        while not math.isclose(test_data.po_flag.value_counts(normalize=True)[1], valid_data.po_flag.value_counts(normalize=True)[1], abs_tol=0.005): # Check to make sure val and test set are comparable
            x +=1
            #if x == 25 or x == 33:
            #    x += 1
            #print('x', x)
            stays = data['stay_id'].unique()
            random.Random(x).shuffle(stays)
            X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
            # Filter for features in this individual
            X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
            model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
            model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
            n = round(0.7 * len(stays))
            n2 = round(0.85 * len(stays))
            train_stays = stays[:n]
            validation_stays = stays[n:n2]
            test_stays = stays[n2:]
            train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
            valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
            test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
            # Oversample train set
            train_data = smote_fun(train_data)

            # Filter for bad bio
            while len(test_data[test_data['stay_id'].isin(sepsis_stays)].po_flag.value_counts(normalize=True)) < 2:
                x +=1
                #if x == 25 or x == 33:
                #    x += 1
                #print('x', x)
                stays = data['stay_id'].unique()
                random.Random(x).shuffle(stays) 
                X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
                # Filter for features in this individual
                X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
                model_data = pd.concat([data[['stay_id', 'po_flag']], X_data], axis=1)
                model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
                n = round(0.7 * len(stays))
                n2 = round(0.85 * len(stays))
                train_stays = stays[:n]
                validation_stays = stays[n:n2]
                test_stays = stays[n2:]
                train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
                valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
                test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
                # Oversample train set
                train_data = smote_fun(train_data)
        
        data_dict[i] = [train_data, valid_data, test_data]
        random_x_list.append(x)

    return data_dict, random_x_list

def test_stats(list1, list2):

    # Test if same distribution
    k2, p = stats.mannwhitneyu(list1, list2)
    alpha = 0.05
    print(p)
    if p < alpha:
        print('Different distribution')
    else:
        print(' Same distribution')

# Function to split data so even distribution between val and test
def los_data_fun(data, individual, n_cv=10, smote_bool=True):
    
    data_dict = {}
    random_x_list = []
    x = -1

    for i in range(n_cv):
        x += 1
        #print('x', x)
        stays = data['stay_id'].unique()
        random.Random(x).shuffle(stays)
        X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag', 'iv_treatment_length'])
        # Filter for features in this individual
        X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
        model_data = pd.concat([data[['stay_id', 'po_flag', 'date', 'iv_treatment_length']], X_data], axis=1)
        model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
        n = round(0.7 * len(stays))
        n2 = round(0.85 * len(stays))
        train_stays = stays[:n]
        validation_stays = stays[n:n2]
        test_stays = stays[n2:]
        train_data = model_data2[model_data2['stay_id'].isin(train_stays)].drop(columns=['date', 'iv_treatment_length'])
        valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)].drop(columns=['date', 'iv_treatment_length'])
        test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
        # Oversample train set
        if smote_bool == True:
            train_data = smote_fun(train_data)
        
        while not math.isclose(test_data.po_flag.value_counts(normalize=True)[1], valid_data.po_flag.value_counts(normalize=True)[1], abs_tol=0.005): # Check to make sure val and test set are comparable
            x +=1
            #print('x', x)
            stays = data['stay_id'].unique()
            random.Random(x).shuffle(stays) 
            X_data = data.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag', 'iv_treatment_length'])
            # Filter for features in this individual
            X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
            model_data = pd.concat([data[['stay_id', 'po_flag', 'date', 'iv_treatment_length']], X_data], axis=1)
            model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
            n = round(0.7 * len(stays))
            n2 = round(0.85 * len(stays))
            train_stays = stays[:n]
            validation_stays = stays[n:n2]
            test_stays = stays[n2:]
            train_data = model_data2[model_data2['stay_id'].isin(train_stays)].drop(columns=['date', 'iv_treatment_length'])
            valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)].drop(columns=['date', 'iv_treatment_length'])
            test_data = model_data2[model_data2['stay_id'].isin(test_stays)]
            # Oversample train set
            if smote_bool == True:
                train_data = smote_fun(train_data)

        data_dict[i] = [train_data, valid_data, test_data]
        random_x_list.append(x)

    return data_dict, random_x_list

def los_run_fun(data_dict):

    icu_stays = pd.read_csv(r"mimic-iv-2.0/icu/icustays.csv")
    icu_stays = icu_stays[['stay_id', 'outtime', 'los']]

    po_los_dict = {}
    iv_los_dict = {}

    # Iterate through dict i.e fold
    for key, value in data_dict.items():
        train_data = value[0]
        valid_data = value[1]
        test_data = value[2]

        test_data = test_data[(test_data['iv_treatment_length'] >= 2) & (test_data['iv_treatment_length'] < 8)]

        for i in test_data.iv_treatment_length.unique():
            #print(i)
            temp_data = test_data[(test_data['iv_treatment_length'] == i) & (test_data['po_flag'] == 1)]
            temp_data = pd.merge(temp_data, icu_stays)
            temp_data['date'] = pd.to_datetime(temp_data['date']).dt.date
            temp_data['outtime'] = pd.to_datetime(temp_data['outtime']).dt.date
            temp_data['remaining_los'] =  (temp_data['outtime'] - temp_data['date']).dt.days
            remaining_los_list = temp_data['remaining_los'].values.tolist()
            if key == 0:
                po_los_dict[i] = remaining_los_list
            else:
                new_los_list = po_los_dict[i]
                new_los_list.extend(remaining_los_list)
                po_los_dict[i] = new_los_list
        
        for i in test_data.iv_treatment_length.unique():
            #print(i)
            temp_data = test_data[(test_data['iv_treatment_length'] == i) & (test_data['po_flag'] == 0)]
            temp_data = pd.merge(temp_data, icu_stays)
            temp_data['date'] = pd.to_datetime(temp_data['date']).dt.date
            temp_data['outtime'] = pd.to_datetime(temp_data['outtime']).dt.date
            temp_data['remaining_los'] =  (temp_data['outtime'] - temp_data['date']).dt.days
            remaining_los_list = temp_data['remaining_los'].values.tolist()
            if key == 0:
                iv_los_dict[i] = remaining_los_list
            else:
                new_los_list = iv_los_dict[i]
                new_los_list.extend(remaining_los_list)
                iv_los_dict[i] = new_los_list
    
    return po_los_dict, iv_los_dict

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

import statistics

def early_late_agree_fun(data_dict, model, string):

    icu_stays = pd.read_csv(r"mimic-iv-2.0/icu/icustays.csv")
    icu_stays = icu_stays[['stay_id', 'hadm_id', 'outtime', 'los']]
    admissions = pd.read_csv(r"mimic-iv-2.0/hosp/admissions.csv")
    admissions = admissions[['hadm_id', 'hospital_expire_flag']]
    icu_stays = pd.merge(icu_stays, admissions)

    # df over epochs
    lb_los = pd.DataFrame()
    ub_los = pd.DataFrame()

    lb_mortality = pd.DataFrame()
    ub_mortality = pd.DataFrame()

    lb_count_df = pd.DataFrame()
    ub_count_df = pd.DataFrame()

    # Lists for averages over epochs
    lb_percentage_agree_list = []
    lb_percentage_late_list = []
    lb_percentage_early_list = []
    ub_percentage_agree_list = []
    ub_percentage_late_list = []
    ub_percentage_early_list = []

    # Define batch size 
    batch_size = 256

    # Define optimizer and learning_rate
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss
    criterion = nn.BCEWithLogitsLoss()

    # Define epochs and clip
    N_EPOCHS = 10 #10
    CLIP = 1

    # Iterate through dict i.e fold
    for key, value in data_dict.items():
        train_data = value[0]
        valid_data = value[1]
        test_data = value[2]

        # Initializing the weights of our model each fold
        model.apply(init_weights)
        
        # Define dataloaders
        train_dataset = MIMICDataset(train_data)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn_padd)

        valid_dataset = MIMICDataset(valid_data)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn_padd)

        test_dataset = MIMICDataset(test_data)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn_padd)

        # Run
        best_valid_loss = float('inf')
        best_valid_auroc = 0

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss, train_accuracy, train_auroc, train_predictions, train_labels = train(model, train_dataloader, optimizer, criterion, CLIP)
            valid_loss, valid_accuracy, valid_auroc, valid_predictions, valid_labels = evaluate(model, valid_dataloader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #print('BEST VALID LOSS')

            if valid_auroc > best_valid_auroc:
                best_valid_auroc = valid_auroc
                #print('UPDATED BEST INTERMEDIATE MODEL')
                torch.save(model.state_dict(), f'hold_out_switch_model_intermediate_{string}.pt')

        # -----------------------------
        # Evaluate best model on los
        # -----------------------------

        model.load_state_dict(torch.load(f'hold_out_switch_model_intermediate_{string}.pt'))

        # Filter for those who switch
        test_stay_id_list = (test_data.groupby(['stay_id'])['po_flag'].nunique() > 1).where(lambda x : x==True).dropna().reset_index()['stay_id'].unique().tolist()
        filtered_test_data = test_data[test_data['stay_id'].isin(test_stay_id_list)]


        # Find the day they actually switched
        test_switch_day = filtered_test_data[filtered_test_data['po_flag'] == 1].drop_duplicates(subset=['stay_id'], keep='first')
        test_switch_day = test_switch_day[['stay_id', 'iv_treatment_length']]
        test_switch_day.rename(columns={'iv_treatment_length': 'real_switch_day'}, inplace=True)
        test_switch_day.reset_index(drop=True, inplace=True)

        # Find LOS and mortality
        icu_stays = pd.merge(icu_stays, admissions)
        test_switch_data = pd.merge(test_switch_day, icu_stays[['stay_id', 'los', 'hospital_expire_flag']])

        # Find remaining LOS - not sure this is apropriate as which day do you choose for remaining LOS calculation. The actual or predicted day?

        # Find day we predict they could switch
        # Get Predictions
        filtered_test_data.reset_index(inplace=True, drop=True)
        filtered_test_data2 = filtered_test_data.drop(columns=['date','iv_treatment_length'])

        # Get predictions
        temp_test_dataset = MIMICDataset(filtered_test_data2)
        temp_test_dataloader = DataLoader(dataset=temp_test_dataset, batch_size=batch_size, collate_fn=temp_test_dataset.collate_fn_padd)

        temp_loss, temp_accuracy, temp_auroc, temp_predictions, temp_labels = evaluate(model, temp_test_dataloader, criterion)
        temp_predictions, ub_temp_predictions = new_threshold_fun(temp_predictions)

        filtered_test_data['lb_prediction'] = temp_predictions
        filtered_test_data['ub_prediction'] = ub_temp_predictions

        # Find the day we predict they switched
        filtered_test_data = lb_predicted_switch_day_fun(filtered_test_data)
        filtered_test_data = ub_predicted_switch_day_fun(filtered_test_data)

        test_lb_predicted_switch_day = filtered_test_data[filtered_test_data['lb_prediction'] == 1].drop_duplicates(subset=['stay_id'], keep='first')
        test_lb_predicted_switch_day = test_lb_predicted_switch_day[['stay_id', 'lb_predicted_switch_day']]
        test_lb_predicted_switch_day.reset_index(drop=True, inplace=True)
        test_lb_predicted_switch_day

        #print(test_lb_predicted_switch_day[test_lb_predicted_switch_day['lb_predicted_switch_day'] == 999])

        test_ub_predicted_switch_day = filtered_test_data[filtered_test_data['ub_prediction'] == 1].drop_duplicates(subset=['stay_id'], keep='first')
        test_ub_predicted_switch_day = test_ub_predicted_switch_day[['stay_id', 'ub_predicted_switch_day']]
        test_ub_predicted_switch_day.reset_index(drop=True, inplace=True)
        test_ub_predicted_switch_day

        #print(test_ub_predicted_switch_day[test_ub_predicted_switch_day['ub_predicted_switch_day'] == 999])

        # Merge and work out difference
        test_switch_data = pd.merge(test_switch_day, icu_stays[['stay_id', 'los', 'hospital_expire_flag']])
        test_switch_data = pd.merge(test_switch_data, test_lb_predicted_switch_day)
        test_switch_data = pd.merge(test_switch_data, test_ub_predicted_switch_day)
        test_switch_data['lb_difference'] = test_switch_data['lb_predicted_switch_day'] - test_switch_data['real_switch_day'] #- test_switch_data['lb_predicted_switch_day']
        test_switch_data['ub_difference'] = test_switch_data['ub_predicted_switch_day'] - test_switch_data['real_switch_day'] #- test_switch_data['ub_predicted_switch_day']

        # Get results
        lb_los_mean = pd.DataFrame(test_switch_data.groupby('lb_difference').los.mean()) 
        lb_mortality_mean = pd.DataFrame(test_switch_data.groupby('lb_difference').hospital_expire_flag.mean())
        lb_count = pd.DataFrame(test_switch_data['lb_difference'].value_counts())

        lb_los_mean.rename(columns={'los':f'los_{key}'}, inplace=True)
        lb_mortality_mean.rename(columns={'hospital_expire_flag':f'mortality_{key}'}, inplace=True)
        lb_count.rename(columns={'lb_difference':f'lb_difference_{key}'}, inplace=True)

        if key == 0:
            lb_los = lb_los_mean
        else:
            #lb_los = pd.merge(lb_los, lb_los_mean, left_index=True, right_index=True)
            lb_los = pd.concat([lb_los, lb_los_mean], axis=1)
        if key == 0:
            lb_mortality = lb_mortality_mean
        else:
            #lb_mortality = pd.merge(lb_mortality, lb_mortality_mean, left_index=True, right_index=True)
            lb_mortality = pd.concat([lb_mortality, lb_mortality_mean], axis=1)
        
        if key == 0:
            lb_count_df = lb_count
        else:
            lb_count_df = pd.concat([lb_count_df, lb_count], axis=1)
        
        lb_percentage_agree = len(test_switch_data[test_switch_data['lb_difference'] == 0])/len(test_switch_data)
        lb_percentage_early = len(test_switch_data[test_switch_data['lb_difference'] < 0])/len(test_switch_data)
        lb_percentage_late = len(test_switch_data[test_switch_data['lb_difference'] > 0])/len(test_switch_data)

        lb_percentage_agree_list.append(lb_percentage_agree)
        lb_percentage_late_list.append(lb_percentage_late)
        lb_percentage_early_list.append(lb_percentage_early)

        ub_los_mean = pd.DataFrame(test_switch_data.groupby('ub_difference').los.mean()) 
        ub_mortality_mean = pd.DataFrame(test_switch_data.groupby('ub_difference').hospital_expire_flag.mean())
        ub_count = pd.DataFrame(test_switch_data['ub_difference'].value_counts())

        ub_los_mean.rename(columns={'los':f'los_{key}'}, inplace=True)
        ub_mortality_mean.rename(columns={'hospital_expire_flag':f'mortality_{key}'}, inplace=True)
        #ub_los = pd.merge(ub_los, ub_los_mean, left_index=True, right_index=True)
        #ub_mortality = pd.merge(ub_mortality, ub_mortality_mean, left_index=True, right_index=True)
        ub_count.rename(columns={'ub_difference':f'ub_difference_{key}'}, inplace=True)

        if key == 0:
            ub_los = ub_los_mean
        else:
            #ub_los = pd.merge(ub_los, ub_los_mean, left_index=True, right_index=True)
            ub_los = pd.concat([ub_los, ub_los_mean], axis=1)
        if key == 0:
            ub_mortality = ub_mortality_mean
        else:
            #ub_mortality = pd.merge(ub_mortality, ub_mortality_mean, left_index=True, right_index=True)
            ub_mortality = pd.concat([ub_mortality, ub_mortality_mean], axis=1)
        
        if key == 0:
            ub_count_df = ub_count
        else:
            ub_count_df = pd.concat([ub_count_df, ub_count], axis=1)
        

        ub_percentage_agree = len(test_switch_data[test_switch_data['ub_difference'] == 0])/len(test_switch_data)
        ub_percentage_early = len(test_switch_data[test_switch_data['ub_difference'] < 0])/len(test_switch_data)
        ub_percentage_late = len(test_switch_data[test_switch_data['ub_difference'] > 0])/len(test_switch_data)

        ub_percentage_agree_list.append(ub_percentage_agree)
        ub_percentage_late_list.append(ub_percentage_late)
        ub_percentage_early_list.append(ub_percentage_early)

        #print(lb_los)
        #print(lb_mortality)
        #print(ub_los)
        #print(ub_mortality)

    #lb
    lb_los_means = pd.DataFrame(lb_los.mean(axis=1))
    lb_los_means.rename(columns={lb_los_means.columns[0]: 'lb_los_mean'}, inplace=True)
    lb_mortality_means = pd.DataFrame(lb_mortality_mean.mean(axis=1))
    lb_mortality_means.rename(columns={lb_mortality_means.columns[0]: 'lb_mortality_means'}, inplace=True)
    lb_count_df2 = pd.DataFrame(lb_count_df.mean(axis=1))
    lb_count_df3 = pd.DataFrame(lb_count_df.sum(axis=1))
    lb_count_df2.rename(columns={lb_count_df2.columns[0]: 'lb_count_sum'}, inplace=True)
    lb_count_df3.rename(columns={lb_count_df3.columns[0]: 'lb_count_sum'}, inplace=True)

    lb_percentage_agree = statistics.mean(lb_percentage_agree_list)
    lb_percentage_late = statistics.mean(lb_percentage_late_list)
    lb_percentage_early = statistics.mean(lb_percentage_early_list)

    #ub
    ub_los_means = pd.DataFrame(ub_los.mean(axis=1))
    ub_los_means.rename(columns={ub_los_means.columns[0]: 'ub_los_mean'}, inplace=True)
    ub_mortality_means = pd.DataFrame(ub_mortality_mean.mean(axis=1))
    ub_mortality_means.rename(columns={ub_mortality_means.columns[0]: 'ub_mortality_means'}, inplace=True)
    ub_count_df2 = pd.DataFrame(ub_count_df.mean(axis=1))
    ub_count_df3 = pd.DataFrame(ub_count_df.sum(axis=1))
    ub_count_df2.rename(columns={ub_count_df2.columns[0]: 'ub_count_mean'}, inplace=True)
    ub_count_df3.rename(columns={ub_count_df3.columns[0]: 'ub_count_sum'}, inplace=True)

    ub_percentage_agree = statistics.mean(ub_percentage_agree_list)
    ub_percentage_late = statistics.mean(ub_percentage_late_list)
    ub_percentage_early = statistics.mean(ub_percentage_early_list)

    return lb_los_means, lb_mortality_means, lb_count_df2, lb_count_df3, lb_percentage_agree, lb_percentage_early, lb_percentage_late, ub_los_means, ub_mortality_means, ub_count_df2, ub_count_df3, ub_percentage_agree, ub_percentage_early, ub_percentage_late

# Similar function to iv_treatment_length_fun - iv_treatment length to prior days treatment length 

# For only having one positive switch day per stay
def lb_predicted_switch_day_fun(data):
    # Convert to datetime
    data['date'] = pd.to_datetime(data['date'])

    # iv_treatment_length
    cumcount = []
    count = 0
    pos = -1
    flag = 0

    for x in range(len(data)):
        pos += 1
        if pos == len(data) - 1:
            cumcount.append(count) # add count to last one
            break # end
        elif pos == 0:
            cumcount.append(count) # add 0 to first one
            count += 1
        elif data.iloc[x]['stay_id'] == data.iloc[x+1]['stay_id']:
            if data.iloc[x]['lb_prediction'] == 0:
                cumcount.append(count)
                count += 1
            elif flag == 1:
                cumcount.append(999)
                count = 0
                flag = 1
            elif data.iloc[x]['stay_id'] != data.iloc[x-1]['stay_id']:
                if data.iloc[x]['lb_prediction'] == 1:
                    cumcount.append(count)
                    count += 1
                else:
                    cumcount.append(999)
                    count = 0
            else:
                cumcount.append(count)
                count = 0
                flag = 1
        else:
            if data.iloc[x]['lb_prediction'] == 0:
                cumcount.append(count)
                count = 0
                flag = 0
            elif flag == 1:
                cumcount.append(999)
                count = 0
                flag = 0
            else:
                cumcount.append(count)
                count = 0
                flag = 0

    #print(cumcount)
    print(len(cumcount))

    data['lb_predicted_switch_day'] = cumcount
    
    return data

    # Similar function to iv_treatment_length_fun - iv_treatment length to prior days treatment length 

# For only having one positive switch day per stay
def ub_predicted_switch_day_fun(data):
    # Convert to datetime
    data['date'] = pd.to_datetime(data['date'])

    # iv_treatment_length
    cumcount = []
    count = 0
    pos = -1
    flag = 0

    for x in range(len(data)):
        pos += 1
        if pos == len(data) - 1:
            cumcount.append(count) # add count to last one
            break # end
        elif pos == 0:
            cumcount.append(count) # add 0 to first one
            count += 1
        elif data.iloc[x]['stay_id'] == data.iloc[x+1]['stay_id']:
            if data.iloc[x]['ub_prediction'] == 0:
                cumcount.append(count)
                count += 1
            elif flag == 1:
                cumcount.append(999)
                count = 0
                flag = 1
            elif data.iloc[x]['stay_id'] != data.iloc[x-1]['stay_id']:
                if data.iloc[x]['ub_prediction'] == 1:
                    cumcount.append(count)
                    count += 1
                else:
                    cumcount.append(999)
                    count = 0
            else:
                cumcount.append(count)
                count = 0
                flag = 1
        else:
            if data.iloc[x]['ub_prediction'] == 0:
                cumcount.append(count)
                count = 0
                flag = 0
            elif flag == 1:
                cumcount.append(999)
                count = 0
                flag = 0
            else:
                cumcount.append(count)
                count = 0
                flag = 0

    #print(cumcount)
    print(len(cumcount))

    data['ub_predicted_switch_day'] = cumcount
    
    return data

# Change iv_treatment length to prior days treatment length 

# For only having one positive switch day per stay
def iv_treatment_length_fun(data):
    # Convert to datetime
    data['date'] = pd.to_datetime(data['date'])

    # iv_treatment_length
    cumcount = []
    count = 0
    pos = -1
    flag = 0

    for x in range(len(data)):
        pos += 1
        if pos == len(data) - 1:
            cumcount.append(999) # add count to last one
            break # end
        elif pos == 0:
            cumcount.append(count) # add 0 to first one
            count += 1
        elif data.iloc[x]['stay_id'] == data.iloc[x+1]['stay_id']:
            if data.iloc[x]['iv_flag'] == 1:
                cumcount.append(count)
                count += 1
            elif flag == 1:
                cumcount.append(999)
                count = 0
                flag = 1
            elif data.iloc[x]['stay_id'] != data.iloc[x-1]['stay_id']:
                cumcount.append(999)
                count = 0
            else:
                cumcount.append(count)
                count = 0
                flag = 1
        else:
            if data.iloc[x]['iv_flag'] == 1:
                cumcount.append(count)
                count = 0
            elif flag == 1:
                cumcount.append(999)
                count = 0
                flag = 0
            else:
                cumcount.append(count)
                count = 0
                flag = 0

    #print(cumcount)
    print(len(cumcount))

    data['iv_treatment_length'] = cumcount
    
    return data