# -*- coding: utf-8 -*-
# Import dependencies
import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from utils import describe_mol, name2feat, matrices_operation, kernel_pair_feat, model_build, export_to_csv


# Constants
DATA_DIR = '../data'
RESULTS_DIR = 'predict_tuna'

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
train_mol_dict = pd.read_csv(f'{DATA_DIR}/eval_chemical_dict.csv')
prior_df = pd.read_csv(f'{DATA_DIR}/prior_data.csv')
eval_df = pd.read_csv(f'{DATA_DIR}/screening_data.csv')
train_df = pd.concat([prior_df, eval_df])
train_df.drop_duplicates(inplace=True)
train_df.reset_index(drop=True, inplace=True)
with open(f'{DATA_DIR}/eval_chemical_feat.pkl', 'rb') as input_file:
    train_mol_feat = pickle.load(input_file)


# Define the core function
def predict_new(test_df, test_mol_dict, train_df=train_df, train_mol_dict=train_mol_dict, train_mol_feat=train_mol_feat):
    # Molecule featurization
    test_mol_feat = {s: describe_mol(s) for s in tqdm(test_mol_dict['smiles'], desc='new molecule featurization')}

    # Model building
    feat_start = time.time()
    X_train, y_train, X_test = kernel_pair_feat(train_df, test_df, train_mol_dict, train_mol_feat, retro=False, test_mol_dict=test_mol_dict, test_mol_feat=test_mol_feat)
    feat_end = time.time()
    feat_duration = feat_end - feat_start
    model = SVC(probability=True, kernel='precomputed')
    output = model_build(model, 'kernel', X_train, y_train, X_test)
    time_tracking = [output[0], feat_duration, output[2], output[3], len(y_train), len(test_df)]

    ## summarize and export results
    test_df['probability'] = output[1]
    time_tracking_df = pd.DataFrame(data=np.column_stack(time_tracking), columns=['model_name', 'feat_duration_s', 'train_duration_s', 'predict_duration_s', 'train_count', 'predict_count'])
    export_to_csv(test_df, RESULTS_DIR, f'{RESULTS_DIR}_result.csv')
    export_to_csv(time_tracking_df, RESULTS_DIR, f'{RESULTS_DIR}_time.csv')
    print('Job done!')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-data', '--test_data', default=None, help='pairs to be predicted, .csv file')
parser.add_argument('-dict', '--test_mol_dict', default=None, help='chemical dictionary mapping names to smiles, .csv file')
args = vars(parser.parse_args())

test_df = pd.read_csv(args['test_data'])
test_mol_dict = pd.read_csv(args['test_mol_dict'])

predict_new(test_df, test_mol_dict)

