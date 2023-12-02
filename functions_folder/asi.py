# Libraries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 16)
#pd.set_option('display.width', 2000)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
pd.options.mode.chained_assignment = None


def calculate_asi(antibiotics_df):

    # Import
    ASI = pd.read_csv('ASI.csv')
    ASI_list = ASI['Short Drug'].to_list()
    ASI2 = ASI.drop(columns=['Drug', 'Antibiotic Spectrum Index']).set_index('Short Drug')

    cumcount = []
    cumcount2 = []

    for x in range(len(antibiotics_df)):
        iv_antibiotic_list = []
        po_antibiotic_list = []
        iv_antibiotic_list = [string for string in ASI_list if string in antibiotics_df.loc[x]['iv_antibiotics'].lower()]
        iv_sub_df = ASI2.loc[iv_antibiotic_list]
        iv_total_df = iv_sub_df.any(axis=0)
        iv_asi_score = iv_total_df.sum()
        cumcount.append(iv_asi_score)
        po_antibiotic_list = [string for string in ASI_list if string in antibiotics_df.loc[x]['po_antibiotics'].lower()]
        po_sub_df = ASI2.loc[po_antibiotic_list]
        po_total_df = po_sub_df.any(axis=0)
        po_asi_score = po_total_df.sum()
        cumcount2.append(po_asi_score)

    antibiotics_df['iv_asi'] = cumcount
    antibiotics_df['po_asi'] = cumcount2

    # % Change
    antibiotics_df['%_change_asi'] = (antibiotics_df.po_asi - antibiotics_df.iv_asi) / antibiotics_df.iv_asi * 100
    # Decrease
    antibiotics_df['decrease_asi'] = np.where(antibiotics_df['po_asi'] < antibiotics_df['iv_asi'], True, False)

    # Mean iv_asi
    print('Mean iv_asi:', antibiotics_df['iv_asi'].mean())
    print('STD iv_asi:', antibiotics_df['iv_asi'].std())
    # Mean po_asi
    print('Mean po_asi:', antibiotics_df['po_asi'].mean())
    print('STD po_asi:', antibiotics_df['po_asi'].std())
    # Mean % ASI change
    print('Mean % ASI change:', antibiotics_df['%_change_asi'].mean())
    print('STD % ASI change:', antibiotics_df['%_change_asi'].std())
    # Percentage decreasing ASI
    print('Percentage who decrease ASI:', (len(antibiotics_df[antibiotics_df['decrease_asi'] == True])/len(antibiotics_df))*100)
    #Mean iv_asi for those who decrease
    print('Mean iv_asi for those who decrease:', antibiotics_df[antibiotics_df['decrease_asi'] == True]['iv_asi'].mean())
    print('STD iv_asi for those who decrease:', antibiotics_df[antibiotics_df['decrease_asi'] == True]['iv_asi'].std())
    # Mean po_asi for those who decrease
    print('Mean po_asi for those who decrease:', antibiotics_df[antibiotics_df['decrease_asi'] == True]['po_asi'].mean())
    print('STD po_asi for those who decrease:', antibiotics_df[antibiotics_df['decrease_asi'] == True]['po_asi'].std())
    # Mean % ASI change for those who decrease 
    print('Mean % ASI change for those who decrease:', antibiotics_df[antibiotics_df['decrease_asi'] == True]['%_change_asi'].mean())
    print('STD % ASI change for those who decrease:', antibiotics_df[antibiotics_df['decrease_asi'] == True]['%_change_asi'].std())

    return antibiotics_df