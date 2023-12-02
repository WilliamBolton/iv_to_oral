from cmath import nan
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
from functions_folder.nn_evaluate import evaluate
from functions_folder.nn_MIMICDataset import MIMICDataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score

def equalised_odds_old(data, batch_size, model, criterion):
    # Import
    admissions = pd.read_csv(r"mimic-iv-2.0/hosp/admissions.csv")
    patients = pd.read_csv(r"mimic-iv-2.0/hosp/patients.csv")
    icu_stays = pd.read_csv(r"mimic-iv-2.0/icu/icustays.csv")
    # Filter for relevant columns 
    admissions = admissions[['subject_id', 'insurance', 'language', 'marital_status', 'race']]
    patients  = patients[['subject_id', 'gender', 'anchor_age']]
    patients['anchor_age'] = (patients['anchor_age'] / 10).round().astype(int) * 10 # Round age to nearest 10
    icu_stays = icu_stays[['stay_id', 'subject_id']]
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

    #print(len(admissions))
    #print(admissions['race'].value_counts())

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
    #print(len(admissions))
    #print(len(new_admissions))
    #print(new_admissions['grouped_race'].value_counts())

    # Get stays
    stay_list = data.stay_id.unique().tolist()
    # Filter for stays 
    icu_stays = icu_stays[icu_stays['stay_id'].isin(stay_list)]
    # Merge
    demographics = icu_stays.merge(patients)
    demographics = demographics.merge(new_admissions)
    demographics.drop(columns=['subject_id'], inplace=True)
    demographics.set_index('stay_id', inplace=True)

    # Fill in nan
    demographics = demographics.fillna('unknown')

    # Blank df
    equalised_odds_df = pd.DataFrame()

    # Iterate through columns and values 
    for column in demographics:
        for value in sorted(demographics[column].unique()):
            sub_demographics = demographics[demographics[column] == value]
            sub_stay_list = sub_demographics.reset_index().stay_id.unique().tolist()
            sub_data = data[data['stay_id'].isin(sub_stay_list)]
            if len(sub_data) > 0: 
                # Define dataloader
                sub_test_dataset = MIMICDataset(sub_data)
                sub_test_dataloader = DataLoader(dataset=sub_test_dataset, batch_size=batch_size, collate_fn=sub_test_dataset.collate_fn_padd)
                # Get results
                test_loss, test_accuracy, test_auroc, test_predictions, test_labels = evaluate(model, sub_test_dataloader, criterion)
                
                # Use new cut off
                lower_bound=0.5427614
                lower_bound_test_predictions = [1 if a_ >= lower_bound else 0 for a_ in test_predictions]

                # Need this as some groups only contain one label
                try:
                    test_auroc = roc_auc_score(test_labels, lower_bound_test_predictions)
                except:
                    test_auroc = np.nan
                try:
                    test_balanced_accuracy = balanced_accuracy_score(test_labels, lower_bound_test_predictions)
                except:
                    test_balanced_accuracy = np.nan

                cm = confusion_matrix(test_labels, lower_bound_test_predictions)
                if cm.shape == (2, 2):
                    #print(value, cm, len(sub_data))
                    true_positive_rate = (cm[0][0]) / (cm[0][0] + cm[0][1])
                    false_positive_rate = (cm[1][0]) / (cm[1][0] + cm[1][1])
                else:
                    true_positive_rate = np.nan
                    false_positive_rate = np.nan
                # Create df
                sub_eo_df = pd.DataFrame([[column, value, test_balanced_accuracy, test_auroc, cm, true_positive_rate, false_positive_rate]])
                #print('sub_eo_df', sub_eo_df)
            else:
                # Create df
                sub_eo_df = pd.DataFrame([[column, value, np.nan, np.nan, np.nan, np.nan, np.nan]])
            # Combine df
            equalised_odds_df = pd.concat([equalised_odds_df, sub_eo_df], axis=0, ignore_index=True)
            #print('equalised_odds_df', equalised_odds_df)
    # Name columns
    equalised_odds_df.columns = ['column', 'value', 'accuracy', 'auroc', 'cm', 'true_positive_rate', 'false_positive_rate']

    return equalised_odds_df

def equalised_odds(data, batch_size, model, criterion, lower_bound_cutoff=0.5427614):
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

    #print(len(admissions))
    #print(admissions['race'].value_counts())

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
    #print(len(admissions))
    #print(len(new_admissions))
    #print(new_admissions['grouped_race'].value_counts())

    # Get stays
    stay_list = data.stay_id.unique().tolist()
    # Filter for stays 
    icu_stays = icu_stays[icu_stays['stay_id'].isin(stay_list)]
    # Merge
    demographics = icu_stays.merge(patients)
    demographics = demographics.merge(new_admissions)
    # All duplicates seem to show this white then other race behaviour so drop other
    demographics.drop_duplicates(subset=['stay_id', 'hadm_id', 'subject_id', 'gender', 'anchor_age', 'insurance', 'language', 'marital_status'], inplace=True)
    demographics.drop(columns=['subject_id', 'hadm_id'], inplace=True)
    demographics.set_index('stay_id', inplace=True)

    # Fill in nan
    demographics = demographics.fillna('unknown')

    # Blank df
    equalised_odds_df = pd.DataFrame()

    # Iterate through columns and values 
    for column in demographics:
        for value in sorted(demographics[column].unique()):
            sub_demographics = demographics[demographics[column] == value]
            sub_stay_list = sub_demographics.reset_index().stay_id.unique().tolist()
            sub_data = data[data['stay_id'].isin(sub_stay_list)]
            if len(sub_data) > 0: 
                # Define dataloader
                sub_test_dataset = MIMICDataset(sub_data)
                sub_test_dataloader = DataLoader(dataset=sub_test_dataset, batch_size=batch_size, collate_fn=sub_test_dataset.collate_fn_padd)
                # Get results
                test_loss, test_accuracy, test_auroc, test_predictions, test_labels = evaluate(model, sub_test_dataloader, criterion)
                
                # Use new cut off
                #lower_bound=0.5427614
                #lower_bound_test_predictions = [1 if a_ >= lower_bound else 0 for a_ in test_predictions]
                lower_bound_test_predictions, upper_bound_test_predictions = eo_new_threshold_fun(test_predictions, lower_bound=lower_bound_cutoff)

                # Need this as some groups only contain one label
                try:
                    test_auroc = roc_auc_score(test_labels, lower_bound_test_predictions)
                except:
                    test_auroc = np.nan
                try:
                    test_balanced_accuracy = balanced_accuracy_score(test_labels, lower_bound_test_predictions)
                except:
                    test_balanced_accuracy = np.nan

                cm = confusion_matrix(test_labels, lower_bound_test_predictions)
                if cm.shape == (2, 2):
                    #print(value, cm, len(sub_data))
                    tn, fp, fn, tp = cm.ravel()
                    true_positive_rate = (tp / (tp + fn))
                    false_positive_rate = (fp / (fp + tn))
                else:
                    true_positive_rate = np.nan
                    false_positive_rate = np.nan
                # Create df
                sub_eo_df = pd.DataFrame([[column, value, test_balanced_accuracy, test_auroc, cm, true_positive_rate, false_positive_rate]])
                #print('sub_eo_df', sub_eo_df)
            else:
                # Create df
                sub_eo_df = pd.DataFrame([[column, value, np.nan, np.nan, np.nan, np.nan, np.nan]])
            # Combine df
            equalised_odds_df = pd.concat([equalised_odds_df, sub_eo_df], axis=0, ignore_index=True)
            #print('equalised_odds_df', equalised_odds_df)
    # Name columns
    equalised_odds_df.columns = ['column', 'value', 'accuracy', 'auroc', 'cm', 'true_positive_rate', 'false_positive_rate']

    return equalised_odds_df

# Test select
def eo_new_threshold_fun(predictions, lower_bound=0.5427614, upper_bound=0.736):
    lower_bound_predictions = [1 if a_ >= lower_bound else 0 for a_ in predictions]
    upper_bound_predictions = [1 if a_ >= upper_bound else 0 for a_ in predictions]
    return lower_bound_predictions, upper_bound_predictions