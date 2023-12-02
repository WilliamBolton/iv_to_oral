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
from datetime import datetime
from datetime import timedelta

def specific_antibiotics(whole_data, mimic=True):

    if mimic == True:
        # Get stays
        whole_stay_list = whole_data.stay_id.unique().tolist()

        # Import
        antibiotics = pd.read_csv(r"mimic-iv-2.0/antibiotic.csv")

        # Drop Mupirocin
        antibiotics = antibiotics[~antibiotics['antibiotic'].str.contains('Mupirocin', case=False)]

        # Filter for stays 
        antibiotics = antibiotics[antibiotics['stay_id'].isin(whole_stay_list)]

        # Filter for relevant delivery methods
        route_list = ['IV', 'PO/NG', 'PO', 'NU', 'ORAL']
        antibiotics['flag'] = np.where(antibiotics.route.str.contains('|'.join(route_list), na=False, case=False),1,0)
        antibiotics = antibiotics[antibiotics['flag'] == 1]
        antibiotics.drop(columns=['flag'], inplace=True)
        # Need to remove some others that got through the filter 
        antibiotics = antibiotics.groupby('route').filter(lambda x: len(x) > 100)
        antibiotics['route'] = antibiotics['route'].replace({'PO/NG':'PO', 'NU':'PO', 'ORAL':'PO'})

        # Create df with date range for IV and PO
        antibiotics2 = antibiotics[['stay_id', 'antibiotic', 'route', 'starttime', 'stoptime']]
        antibiotics2.dropna(inplace=True)
        # Remove hours from dates
        antibiotics2['starttime'] = pd.to_datetime(antibiotics2['starttime']).dt.date
        antibiotics2['stoptime'] = pd.to_datetime(antibiotics2['stoptime']).dt.date
        # Create column of antibiotic and route
        antibiotics2['antibiotic_route'] = antibiotics2[['antibiotic', 'route']].agg('_'.join, axis=1)
        antibiotics2['date'] = antibiotics2.apply(lambda x:
            pd.date_range(start=x['starttime'],
                        end=x['stoptime'],
                        #inclusive='both',
                        freq='D'), axis=1)
        antibiotics3 = antibiotics2.explode('date')

        # Groupby
        groupby = ['stay_id',
                'date']

        # Create daily therapies
        antibiotics4 = antibiotics3.groupby(groupby) \
            .apply(lambda x: sorted(x.antibiotic_route \
                .unique()))

        # To df
        antibiotics4 = antibiotics4.to_frame()
        antibiotics4 = antibiotics4.reset_index()
        antibiotics4 = antibiotics4.rename(columns= {0: 'antibiotics'})
        antibiotics4['date'] = pd.to_datetime(antibiotics4['date'])
        whole_data['date'] = pd.to_datetime(whole_data['date'])

        # Merge
        antibiotics_df = whole_data[['stay_id', 'date', 'iv_flag', 'first_po_flag', 'po_flag']].merge(antibiotics4)

        # Filter for day before and after change
        index_list = []

        for x in range(len(antibiotics_df)):
            if antibiotics_df.iloc[x]['po_flag'] == 0:
                if x == len(antibiotics_df)-1:
                    pass
                else:
                    if antibiotics_df.iloc[x+1]['po_flag'] == 1:
                        if antibiotics_df.iloc[x]['stay_id'] == antibiotics_df.iloc[x+1]['stay_id']:
                            index_list.append(x)
                            index_list.append(x+1)

        filtered_antibiotics_df = antibiotics_df.iloc[index_list]

        # Need to convert colun of lists to strings or later on wont work
        filtered_antibiotics_df['antibiotics'] = [','.join(map(str, l)) for l in filtered_antibiotics_df['antibiotics']]

        # Create df with column for iv and po
        filtered_antibiotics_iv = filtered_antibiotics_df[filtered_antibiotics_df['po_flag'] == 0]
        filtered_antibiotics_po = filtered_antibiotics_df[filtered_antibiotics_df['po_flag'] == 1]
        filtered_antibiotics_iv.rename(columns={'antibiotics':'iv_antibiotics'}, inplace=True)
        filtered_antibiotics_po.rename(columns={'antibiotics':'po_antibiotics'}, inplace=True)
        filtered_antibiotics_iv.drop(columns=['iv_flag', 'first_po_flag', 'date', 'po_flag'], inplace=True)
        filtered_antibiotics_po.drop(columns=['iv_flag', 'first_po_flag', 'date', 'po_flag'], inplace=True)
        filtered_antibiotics_df2 = pd.merge(filtered_antibiotics_iv, filtered_antibiotics_po, on=['stay_id'])

        antibiotics_value_counts = filtered_antibiotics_df2[['iv_antibiotics', 'po_antibiotics']].value_counts()
    
    elif mimic == False:
        # Get stays
        whole_stay_list = whole_data.stay_id.unique().tolist()

        # Load potential antibiotics 
        infection_treatment_categories = pd.read_csv(r"eicu-collaborative-research-database-2.0/infection_treatment_categories.csv")
        infection_treatment_categories2 = infection_treatment_categories.rx.str.split(pat='|', expand=True)
        infection_treatment_categories2.rename(columns={0:'a', 1:'b', 2:'c', 3:'d', 4:'e'}, inplace=True)
        infection_treatment_categories3 = infection_treatment_categories2.d.str.split(pat='/', expand=True)
        # Create lists of antibiotics 
        list1 = infection_treatment_categories2.d.to_list()
        list2 = infection_treatment_categories2.e.to_list()
        # Add: 'linezolid', 'Zyvox', 'Synercid', 'quinupristin', 'dalfopristin', 'quinupristin/dalfopristin', 'cephalosporin', 'ticarcillin', 'amoxicillin', 'penicillin', 'benzathine', 'piperacillin', 'ampicillin', 
        # Worked these out by splitting other sections
        list3 = ['linezolid', 'Zyvox', 'Synercid', 'quinupristin', 'dalfopristin', 'quinupristin/dalfopristin', 'cephalosporin', 'ticarcillin', 'amoxicillin', 'penicillin', 'benzathine', 'piperacillin', 'ampicillin']
        antibiotic_list = list1 + list2 + list3
        # Drop duplicates through set
        antibiotic_list = list(set(antibiotic_list))
        # Drop None
        antibiotic_list = [x for x in antibiotic_list if x is not None]

        # Load all medications
        medication = pd.read_csv(r"eicu-collaborative-research-database-2.0/medication.csv", dtype={'drugname': 'object'})
        medication = medication[['medicationid', 'patientunitstayid', 'drugstartoffset', 'drugname', 'routeadmin', 'drugivadmixture', 'drugordercancelled', 'drugstopoffset']]
        # Filter for antibiotics 
        medication['flag'] = np.where(medication.drugname.str.contains('|'.join(antibiotic_list), na=False, case=False),1,0)
        antibiotics = medication[medication['flag'] == 1]
        antibiotics = antibiotics[antibiotics['drugordercancelled'] == 'No']
        assert len(antibiotics[antibiotics['drugordercancelled'] == 'No']) == len(antibiotics)
        #print(antibiotics.patientunitstayid.nunique())
        # Set those where 'drugivadmixture' == 'Yes' to 'IV' route
        antibiotics.loc[antibiotics['drugivadmixture'] == 'Yes', 'routeadmin'] = 'IV'
        # Filter for relevant delivery methods
        route_list = ['IV', 'Intrav', 'PO', 'tube', 'ORAL']
        antibiotics = antibiotics[antibiotics.routeadmin.str.contains('|'.join(route_list), na=False, case=False)]
        # Group so all IV or PO
        iv_route_list = ['IV', 'Intrav']
        po_route_list = ['PO', 'tube', 'ORAL']
        antibiotics.loc[antibiotics.routeadmin.str.contains('|'.join(iv_route_list), na=False, case=False), 'routeadmin'] = 'IV'
        antibiotics.loc[antibiotics.routeadmin.str.contains('|'.join(po_route_list), na=False, case=False), 'routeadmin'] = 'PO'
        antibiotics = antibiotics.drop(columns=['drugivadmixture', 'drugordercancelled', 'flag'])
        # Load patients data
        patients = pd.read_csv(r"eicu-collaborative-research-database-2.0/patient.csv")
        # Select relevant columns
        patients2 = patients[['patientunitstayid', 'hospitaldischargestatus', 'unitadmittime24', 'unitdischargeoffset', 'unitdischargestatus']]
        # Filter for those who survived
        patients2 = patients2[(patients2['hospitaldischargestatus'] == 'Alive') & (patients2['unitdischargestatus'] == 'Alive')]
        patients2 = patients2.drop(columns=['hospitaldischargestatus', 'unitdischargestatus'])
        # Merge
        antibiotic_patients = pd.merge(antibiotics, patients2, how="left", on='patientunitstayid')
        # Convert unit admit time to day (day 0)
        antibiotic_patients['unitadmittime24'] = pd.to_datetime(antibiotic_patients['unitadmittime24'])
        # Set start date
        antibiotic_patients['unitadmittime24'] = antibiotic_patients['unitadmittime24'].apply(lambda t: t.replace(year=2022, month=9, day=2))
        # Create starttime and stoptime and discharge time
        antibiotic_patients['drugstartoffset'] = pd.to_timedelta(antibiotic_patients['drugstartoffset'], unit='min') # Convert to timedelta
        antibiotic_patients['drugstopoffset'] = pd.to_timedelta(antibiotic_patients['drugstopoffset'], unit='min') # Convert to timedelta
        antibiotic_patients['unitdischargeoffset'] = pd.to_timedelta(antibiotic_patients['unitdischargeoffset'], unit='min')

        antibiotic_patients['starttime'] = antibiotic_patients['unitadmittime24'] + antibiotic_patients['drugstartoffset']
        antibiotic_patients['stoptime'] = antibiotic_patients['unitadmittime24'] + antibiotic_patients['drugstopoffset']
        antibiotic_patients['dischargetime'] = antibiotic_patients['unitadmittime24'] + antibiotic_patients['unitdischargeoffset']
        # Rename
        antibiotic_patients = antibiotic_patients.rename(columns={'patientunitstayid': 'stay_id', 'routeadmin': 'route', 'drugname': 'antibiotic'})

        # Now same as mimic
        # Filter for stays 
        antibiotics = antibiotic_patients[antibiotic_patients['stay_id'].isin(whole_stay_list)]

         # Create df with date range for IV and PO
        antibiotics2 = antibiotics[['stay_id', 'antibiotic', 'route', 'starttime', 'stoptime']]
        antibiotics2.dropna(inplace=True)
        # Remove hours from dates
        antibiotics2['starttime'] = pd.to_datetime(antibiotics2['starttime']).dt.date
        antibiotics2['stoptime'] = pd.to_datetime(antibiotics2['stoptime']).dt.date
        # Create column of antibiotic and route
        antibiotics2['antibiotic_route'] = antibiotics2[['antibiotic', 'route']].agg('_'.join, axis=1)
        antibiotics2['date'] = antibiotics2.apply(lambda x:
            pd.date_range(start=x['starttime'],
                        end=x['stoptime'],
                        #inclusive='both',
                        freq='D'), axis=1)
        antibiotics3 = antibiotics2.explode('date')

        # Groupby
        groupby = ['stay_id',
                'date']

        # Create daily therapies
        antibiotics4 = antibiotics3.groupby(groupby) \
            .apply(lambda x: sorted(x.antibiotic_route \
                .unique()))

        # To df
        antibiotics4 = antibiotics4.to_frame()
        antibiotics4 = antibiotics4.reset_index()
        antibiotics4 = antibiotics4.rename(columns= {0: 'antibiotics'})
        antibiotics4['date'] = pd.to_datetime(antibiotics4['date'])
        whole_data['date'] = pd.to_datetime(whole_data['date'])

        # Merge
        antibiotics_df = whole_data[['stay_id', 'date', 'iv_flag', 'first_po_flag', 'po_flag']].merge(antibiotics4)

        # Filter for day before and after change
        index_list = []

        for x in range(len(antibiotics_df)):
            if antibiotics_df.iloc[x]['po_flag'] == 0:
                if x == len(antibiotics_df)-1:
                    pass
                else:
                    if antibiotics_df.iloc[x+1]['po_flag'] == 1:
                        if antibiotics_df.iloc[x]['stay_id'] == antibiotics_df.iloc[x+1]['stay_id']:
                            index_list.append(x)
                            index_list.append(x+1)

        filtered_antibiotics_df = antibiotics_df.iloc[index_list]

        # Need to convert colun of lists to strings or later on wont work
        filtered_antibiotics_df['antibiotics'] = [','.join(map(str, l)) for l in filtered_antibiotics_df['antibiotics']]

        # Create df with column for iv and po
        filtered_antibiotics_iv = filtered_antibiotics_df[filtered_antibiotics_df['po_flag'] == 0]
        filtered_antibiotics_po = filtered_antibiotics_df[filtered_antibiotics_df['po_flag'] == 1]
        filtered_antibiotics_iv.rename(columns={'antibiotics':'iv_antibiotics'}, inplace=True)
        filtered_antibiotics_po.rename(columns={'antibiotics':'po_antibiotics'}, inplace=True)
        filtered_antibiotics_iv.drop(columns=['iv_flag', 'first_po_flag', 'date', 'po_flag'], inplace=True)
        filtered_antibiotics_po.drop(columns=['iv_flag', 'first_po_flag', 'date', 'po_flag'], inplace=True)
        filtered_antibiotics_df2 = pd.merge(filtered_antibiotics_iv, filtered_antibiotics_po, on=['stay_id'])

        antibiotics_value_counts = filtered_antibiotics_df2[['iv_antibiotics', 'po_antibiotics']].value_counts()


    return filtered_antibiotics_df2, antibiotics_value_counts