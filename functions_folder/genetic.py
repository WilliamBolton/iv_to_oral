import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import time
import copy
import random
import sklearn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from functions_folder.nn_MIMICDataset import MIMICDataset
from functions_folder.nn_train import train
from functions_folder.nn_evaluate import evaluate
from typing import Counter
import imblearn
from imblearn.over_sampling import SMOTE

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

from functions_folder.nn_model_sequential import Model_sequential

def generate_dataframes_for_training(dataframe, individual):
    """
    Generates training and testing dataframes from a complete dataframe, according to the split_frac parameter
    """
    stays = dataframe['stay_id'].unique()
    import random
    random.Random(5).shuffle(stays) # Note changed random split to 5 so split between po_flag is similar for val and test sets 
    X_data = dataframe.drop(columns=['stay_id', 'po_flag', 'date', 'iv_flag', 'first_po_flag'])
    # Filter for features in this individual
    X_data = X_data.loc[:, [True if individual[i] == '1' else False for i in range(len(individual))]]
    model_data = pd.concat([dataframe[['stay_id', 'po_flag']], X_data], axis=1)
    model_data2 = model_data.set_index("stay_id").loc[stays].reset_index()
    n = round(0.7 * len(stays))
    n2 = round(0.85 * len(stays))
    train_stays = stays[:n]
    validation_stays = stays[n:n2]
    test_stays = stays[n2:]
    train_data = model_data2[model_data2['stay_id'].isin(train_stays)]
    valid_data = model_data2[model_data2['stay_id'].isin(validation_stays)]
    test_data = model_data2[model_data2['stay_id'].isin(test_stays)]

    # Oversampling train set
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
    
    return train_data, valid_data, test_data

def get_n_individual(counter, population):
    """
    If counter is 0, return the individual with the highest prob
    If counter is 1, return the second individual with the highest prob
    If counter is 2, return the third individual withthe highest prob
    """
    index = counter + 1
    probabilities = [ind[1] for ind in population]
    sorted_probs = sorted(probabilities, key=float)
    max_prob = probabilities[-index]
    max_individual = [ind[0] for ind in population if ind[1] == max_prob][0]
    
    return max_individual

def generate_random_individuals(num_individuals, num_features, max_features, verbose=False):
    """
    Randomly generates individuals

    The number of individuals to generate is given by the num_individuals parameter
    The length of each individual is equal to the num_features parameter
    The maximum number of active features for every individual is given by the max_features parameter
    """
    if verbose: print('GENERATING RANDOM INDIVIDUALS.... ')
        
    individuals = list()
    
    for _ in range(num_individuals):
        individual = ''
        for col in range(num_features):
            #print(max_features)
            # For each char in the individual, a 1 or a 0 is randomly generated
            if individual.count('1') == max_features:
                #print('hi')
                individual += '0'
                continue
            #print('here')
            if col == 0:
                individual += str(1) # Set first feature (i.e., treatment length to allways 1 given it is so important)
            else:
                individual += str(random.randint(0, 1))
            
        #if verbose: print(f'Genrated a new indivudal: {individual}')
        individuals.append(individual)
        
    if verbose: print(f'Generated list of {num_individuals} individuals')#: {individuals}')
        
    return individuals

def get_weights(population):
    """
    Calculate weights from the population filled with the auroc
    """
    total_auroc = 0
    new_population = []
    
    # Get the sum of all auroc of the population
    for individual in population:
        total_auroc += individual[1]
        
    # For each individual, calculate its weight by dividing its auroc by the overall sum calculated above
    for individual in population:
        weight = individual[1]/total_auroc
        # Store the individual and its weight in the final population list
        new_population.append((individual[0], float(weight*100)))
        
    return new_population



def get_fitness_func(individual, dataframe, max_features, verbose=False):
    """
    Calculate auroc for the individual passed as parameter.
    Both the dataframe and the y_data parameters are used for training and evaluating the model.
    """
    if verbose: print('Calculating auroc for individual ', individual)
    
    # generate_dataframes_for_training
    train_data, valid_data, test_data = generate_dataframes_for_training(dataframe, individual)  
    #print(train_data)
    #print(valid_data)
    #print(test_data)
    
    INPUT_DIM = len(train_data.columns)-2
    OUTPUT_DIM = 1
    HID_DIM = 128 #128 #768
    HID_DIM2 = 128 #128 #256
    HID_DIM3 = 64
    HID_DIM4 = 8
    DROPOUT = 0.3

    # Define model
    #model = create_model(train_data)
    model = Model_sequential(INPUT_DIM, OUTPUT_DIM, HID_DIM, HID_DIM2, HID_DIM3, HID_DIM4, DROPOUT).to(device)
    model.apply(init_weights)

    batch_size = 256

    train_dataset = MIMICDataset(train_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn_padd)

    valid_dataset = MIMICDataset(valid_data)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn_padd)

    test_dataset = MIMICDataset(test_data)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn_padd)

    eicu_data = pd.read_csv('/home/wb1115/VSCode_projects/stop/csv/eicu_renewed_data.csv')
    #eicu_test_data = eicu_data.drop(columns=['date', 'iv_flag', 'first_po_flag'])
    eicu_test_data = eicu_data[[c for c in eicu_data.columns if c in list(test_data)]]
    eicu_test_dataset = MIMICDataset(eicu_test_data)
    eicu_test_dataloader = DataLoader(dataset=eicu_test_dataset, batch_size=batch_size, collate_fn=eicu_test_dataset.collate_fn_padd)

    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Define epochs and clip
    N_EPOCHS = 10 #10
    CLIP = 1

    # Run
    best_valid_auroc = 0

    for epoch in range(N_EPOCHS):

        train_loss, train_accuracy, train_auroc, train_predictions, train_labels = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss, valid_accuracy, valid_auroc, valid_predictions, valid_labels = evaluate(model, valid_dataloader, criterion)

        if valid_auroc > best_valid_auroc:
            best_valid_auroc = valid_auroc
            #print('UPDATED BEST MODEL')
            torch.save(model.state_dict(), f'intermediate_genetic_model_{max_features}.pt')
        
        #print('Train accuracy result:', train_accuracy)
        #print('Train AUROC result:', train_auroc)
        #print('Validation accuracy result:', valid_accuracy)
        #print('Validation AUROC result:', valid_auroc)

    # -----------------------------
    # Evaluate best model on test set
    # -----------------------------
 
    model.load_state_dict(torch.load(f'intermediate_genetic_model_{max_features}.pt'))

    test_loss, test_accuracy, test_auroc, test_predictions, test_labels = evaluate(model, test_dataloader, criterion)
    print('mimic test AUROC result:', test_auroc)
    eicu_test_loss, eicu_test_accuracy, eicu_test_auroc, eicu_test_predictions, eicu_test_labels = evaluate(model, eicu_test_dataloader, criterion)
    print('eicu Test AUROC result:', eicu_test_auroc)

    if verbose: print(f"auroc for the classifier trained for individual {individual}: ", test_auroc)
    
    return test_auroc

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def fill_population(individuals, dataframe, max_features, verbose=False):
    """
    Fills the population list with individuals and their weights
    """
    population = list()
    
    for individual in individuals:
        
        # Get the value of the fitness function (auroc of the model)
        #if verbose: print(f'Calculating fitness function value for individual {individual}')
        auroc = get_fitness_func(individual, dataframe, max_features, verbose)
        
        # Check that the value is not the goal state (in this case, an auroc of 80% is a terminal state)
        #if float(auroc) > 0.8:
        #    if verbose: print(f'Goal state found for individual {individual}')
        #    return individual
            
        individual_complete = (individual, auroc)
        population.append(individual_complete)
        
    # The final population list is created, which contains each individual together with its weight
    # (weights will be used in the reproduction step)
    new_population = get_weights(population)
    if verbose: print(f'Generated population list (with weights): {new_population}')
    
    return new_population, population



def choose_parents(population, counter):
    """
    From the population, weighting the probabilities of an individual being chosen via the fitness
    function, takes randomly two individual to reproduce
    Population is a list of tuples, where the first element is the individual and the second
    one is the probability associated to it.
    To avoid generating repeated individuals, 'counter' parameter is used to pick parents in different ways, thus
    generating different individuals
    """
    # Pick random parent Number 1 and Number 2
    # (get_n_individual() function randomly picks an individual following the distribution of the weights)
    if counter == 0:        
        parent_1 = get_n_individual(0, population)        
        parent_2 = get_n_individual(1, population)
    elif counter == 1:
        parent_1 = get_n_individual(0, population)        
        parent_2 = get_n_individual(2, population)
        
    else:
        probabilities = (individual[1] for individual in population)
        individuals = [individual[0] for individual in population]
        parent_1, parent_2 = random.choices(individuals, weights=probabilities, k=2)
    
    return [parent_1, parent_2]


def mutate(child, prob=0.1):
    """
    Randomly mutates an individual according to the probability given by prob parameter
    """
    new_child = copy.deepcopy(child)
    for i, char in enumerate(new_child):
        if random.random() < prob:
            new_value = '1' if char == '0' else '0'
            new_child = new_child[:i] + new_value + new_child[i+1:]
    
    return new_child


  
def reproduce(individual_1, individual_2, mutate_bool=True):
    """
    Takes 2 individuals, and combines their information based on a
    randomly chosen crosspoint.
    Each reproduction returns 2 new individuals
    """ 
    # Randomly generate a integer between 1 and the length of the individuals minus one, which will be the crosspoint
    crosspoint = random.randint(1, len(individual_1)-1)
    child_1 = individual_1[:crosspoint] + individual_2[crosspoint:]
    child_2 = individual_2[:crosspoint] + individual_1[crosspoint:]
    if mutate_bool == True:
        child_1, child_2 = mutate(child_1), mutate(child_2)
 
    return [child_1, child_2]


  
def generation_ahead(population, verbose=False, mutate_bool=True):
    """
    Reproduces all the steps for choosing parents and making 
    childs, which means creating a new generation to iterate with
    """
    new_population = list()
    
    for _ in range(int(len(population)//2)):      
        # According to the weights calculated before, choose a set of parents to reproduce
        parents = choose_parents(population, counter=_)
        if verbose: print(f'Parents chosen: {parents}')
          
        # Reproduce the pair of individuals chose above to generate two new individuals
        childs = reproduce(parents[0], parents[1], mutate_bool)
        if verbose: print(f'Generated children: {childs}\n')
        new_population += childs
        
    return new_population



def main_genetic_loop(ind_num, dataframe, max_iter=5, max_features=5, mutate_bool=True, verbose=False):
    """
    Performs all the steps of the Genetic Algorithm
    1. Generate random population
    2. Fill population with the weights of each individual
    3. Check if the goal state is reached
    4. Reproduce the population, and create a new generation
    5. Repeat process until termination condition is met
    """
    iteration_count = 0
    if verbose: print(f'\n\n\nITERATION NUMBER {iteration_count+1} (Iteration max = {max_iter})\n\n\n')

    # Generate individuals (returns a list of strings, where each str represents an individual)
    #individuals = generate_random_individuals(ind_num, len(dataframe.columns)-5, len(dataframe.columns)-5, verbose)
    individuals = generate_random_individuals(ind_num, len(dataframe.columns)-5, max_features, verbose)

    # Returns a list of tuples, where each tuple represents an individual and its weight
    population, population_auroc  = fill_population(individuals, dataframe, max_features, verbose)
    whole_population = {}
    whole_population['1'] = population_auroc
    
    # Check if a goal state is reached
    # When goal state is reached, fill_population() function returns a str, otherwise continue
    #if isinstance(population, str):
    #    return population
        
    # Reproduce current generation to generate a better new one
    new_generation = generation_ahead(population, verbose, mutate_bool)
    
    # After the new generation is generated, the loop goes on until a solution is found or until the maximum number of
    # iterations are reached
    iteration_count = 1
    while iteration_count < max_iter:
        if verbose: print(f'\n\n\nITERATION NUMBER {iteration_count+1} (Iteration max = {max_iter})\n\n\n')
        population, population_auroc = fill_population(new_generation, dataframe, max_features, verbose)
        #print('hi')
        whole_population[str(iteration_count+1)] = population_auroc

        
        # Check if a goal state is reached
        #if isinstance(population, str):
        #    break
        
        new_generation = generation_ahead(population, verbose)   
        iteration_count += 1
        
    return whole_population