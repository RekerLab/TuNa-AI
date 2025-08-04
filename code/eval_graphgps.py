#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import pandas as pd
from tqdm import tqdm

# Constants
DATA_DIR = '../data'
RESULTS_DIR = 'eval_graphgps'

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
prior_df = pd.read_csv(f'{DATA_DIR}/prior_data.csv')
eval_df = pd.read_csv(f'{DATA_DIR}/screening_data.csv')
mol_dict_df = pd.read_csv(f'{DATA_DIR}/eval_chemical_dict.csv')
mol_dict = dict(zip(mol_dict_df['name'], mol_dict_df['smiles']))


def prepare_data(train_df, test_df, mol_dict):
    """
    Add SMILES columns to train and test DataFrames based on molecule names.
    """
    for df in (train_df, test_df):
        df['drug_smiles'] = df['drug_name'].map(mol_dict)
        df['excp_smiles'] = df['excp_name'].map(mol_dict)


def save_dataframes(wd, train_df, test_df):
    """
    Save train and test DataFrames to the working directory.
    """
    os.makedirs(wd, exist_ok=True)
    train_df.to_csv(f'{wd}/train.csv', index=False)
    test_df.to_csv(f'{wd}/test.csv', index=False)


def build_model(wd):
    """
    Run chemprop training with given working directory.
    """
    train_path = f'{wd}/train.csv'
    test_path = f'{wd}/test.csv'
    save_dir = wd

    args = [
        'graphgps_train',
        '--cfg_file', './GPS_classification.yml',
        '--smiles_columns', 'drug_smiles', 'excp_smiles',
        '--features_columns', 'excp_drug_ratio',
        '--target', 'binary',
        '--save_dir', save_dir,
        '--data_path', train_path,
        '--separate_test_path', test_path
    ]

    subprocess.run(args, check=True)

    # Read model prediction output and append probabilities to test data
    raw_output_path = f'{wd}/0/test/pred_best.csv'
    pred_df = pd.read_csv(raw_output_path)
    proba = pred_df['pred'].tolist()

    test_df = pd.read_csv(test_path)
    test_df['graphgps_proba'] = proba
    test_df.to_csv(f'{wd}/output.csv', index=False)


def eval_pipeline(mode, prior_df=prior_df, eval_df=eval_df, mol_dict=mol_dict):
    """
    Evaluate model in one of three modes: 'lodo', 'loeo', or 'lopo'.
    """
    if mode in ['lodo', 'loeo']:
        col_name = 'drug_name' if mode == 'lodo' else 'excp_name'
        unique_molecules = eval_df[col_name].unique()

        for mol in tqdm(unique_molecules, desc=f"Processing {mode}"):
            wd = f'{RESULTS_DIR}/{mode}_{mol}'

            # Skip if output exists
            if os.path.exists(f'{wd}/output.csv'):
                continue

            train_df = pd.concat(
                [prior_df, eval_df[eval_df[col_name] != mol]], ignore_index=True
            )
            test_df = eval_df[eval_df[col_name] == mol].copy()

            if test_df.empty:
                continue

            prepare_data(train_df, test_df, mol_dict)
            save_dataframes(wd, train_df, test_df)
            build_model(wd)

    elif mode == 'lopo':
        unique_pairs = set(zip(eval_df['drug_name'], eval_df['excp_name']))

        for drug, excp in tqdm(unique_pairs, desc="Processing lopo"):
            wd = f'{RESULTS_DIR}/{mode}_{drug}_{excp}'

            if os.path.exists(f'{wd}/output.csv'):
                continue

            test_df = eval_df[
                (eval_df['drug_name'] == drug) & (eval_df['excp_name'] == excp)
            ].copy()

            if test_df.empty:
                continue

            train_df = pd.concat([
                prior_df,
                eval_df[~((eval_df['drug_name'] == drug) & (eval_df['excp_name'] == excp))]
            ], ignore_index=True)

            prepare_data(train_df, test_df, mol_dict)
            save_dataframes(wd, train_df, test_df)
            build_model(wd)

    else:
        raise KeyError("Mode must be one of: 'lodo', 'loeo', or 'lopo'.")


# Run evaluation
for mode in ['lodo', 'loeo', 'lopo']:
    eval_pipeline(mode)
