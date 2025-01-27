# -*- coding: utf-8 -*-
# Import dependencies
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import chemprop

# Constants
DATA_DIR = '../data'
RESULTS_DIR = 'eval_chemprop'

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
prior_df = pd.read_csv(f'{DATA_DIR}/prior_data.csv')
eval_df = pd.read_csv(f'{DATA_DIR}/screening_data.csv')
mol_dict_df = pd.read_csv(f'{DATA_DIR}/eval_chemical_dict.csv')

# Define core functions
mol_dict = dict(zip(mol_dict_df['name'], mol_dict_df['smiles']))

def prepare_data(train_df, test_df, mol_dict):
    """
    Adds SMILES columns to train and test dataframes based on molecule names.
    """
    for df in [train_df, test_df]:
        df['drug_smiles'] = df['drug_name'].map(mol_dict)
        df['excp_smiles'] = df['excp_name'].map(mol_dict)

def save_dataframes_and_ratios(wd, train_df, test_df):
    """
    Saves train/test data and ratios to the specified directory.
    """
    os.makedirs(wd, exist_ok=True)

    # Save train data and ratio
    train_df.to_csv(f'{wd}/train.csv', index=False)
    train_df[['excp_drug_ratio']].to_csv(f'{wd}/train_ratio.csv', index=False)

    # Save test data and ratio
    test_df.to_csv(f'{wd}/test.csv', index=False)
    test_df[['excp_drug_ratio']].to_csv(f'{wd}/test_ratio.csv', index=False)

def train_model(wd):
    """
    Trains a chemprop model with the specified directory's train data and ratios.
    """
    arguments = [
        '--data_path', f'{wd}/train.csv',
        '--dataset_type', 'classification',
        '--save_dir', wd,
        '--number_of_molecules', '2',
        '--smiles_columns', 'drug_smiles', 'excp_smiles',
        '--target_columns', 'class',
        '--features_path', f'{wd}/train_ratio.csv'
    ]
    args = chemprop.args.TrainArgs().parse_args(arguments)
    chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

def evaluate_model(wd):
    """
    Evaluates a chemprop model using the specified directory's test data and ratios.
    """
    arguments = [
        '--test_path', f'{wd}/test.csv',
        '--checkpoint_dir', wd,
        '--preds_path', f'{wd}/output.csv',
        '--number_of_molecules', '2',
        '--smiles_columns', 'drug_smiles', 'excp_smiles',
        '--features_path', f'{wd}/test_ratio.csv'
    ]
    args = chemprop.args.PredictArgs().parse_args(arguments)
    chemprop.train.make_predictions(args=args)

def eval_pipeline(mode, prior_df=prior_df, eval_df=eval_df, mol_dict=mol_dict):
    if mode in ['lodo', 'loeo']:
        col_name = 'drug_name' if mode == 'lodo' else 'excp_name'
        unique_mol = eval_df[col_name].unique()
        for mol in tqdm(unique_mol, desc=f"Processing {mode}"):
            # Set working directory for the current molecule
            wd = f'{RESULTS_DIR}/{mode}_{mol}'
            # Split data into train and test
            train_df = pd.concat([prior_df, eval_df])
            train_df = train_df[train_df[col_name] != mol].copy()
            test_df = eval_df[eval_df[col_name] == mol].copy()
            # Prepare data with SMILES columns
            prepare_data(train_df, test_df, mol_dict)
            # Save dataframes and ratios
            save_dataframes_and_ratios(wd, train_df, test_df)
            # Train and evaluate the model
            train_model(wd)
            evaluate_model(wd)

    elif mode == 'lopo':
        unique_pair = list(set([tuple((eval_df['drug_name'][i], eval_df['excp_name'][i])) for i in range(len(eval_df))]))
        for pair in tqdm(unique_pair, desc=mode):
            # Set working directory for the current molecule
            wd = f'{RESULTS_DIR}/{mode}_{pair}'     
            # Split data into train and test
            train_df = pd.concat([prior_df, eval_df])
            train_df = train_df[(train_df['drug_name'] != pair[0]) & (train_df['excp_name'] != pair[1])].copy()
            test_df = eval_df[(eval_df['drug_name'] == pair[0]) & (eval_df['excp_name'] == pair[1])].copy()
            # Prepare data with SMILES columns
            prepare_data(train_df, test_df, mol_dict)
            # Save dataframes and ratios
            save_dataframes_and_ratios(wd, train_df, test_df)
            # Train and evaluate the model
            train_model(wd)
            evaluate_model(wd)

for mode in ['lodo', 'loeo', 'lopo']:
    eval_pipeline(mode)

