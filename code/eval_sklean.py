# -*- coding: utf-8 -*-
# Import dependencies
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from utils import name2feat, matrices_operation, ctrl_pair_feat, kernel_pair_feat, sim2dis, model_build, export_to_csv

# Load supervised machine learning algorithms
import xgboost
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel

# Constants
DATA_DIR = '../data'
RESULTS_DIR = 'eval_sklean'

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
prior_df = pd.read_csv(f'{DATA_DIR}/prior_data.csv')
eval_df = pd.read_csv(f'{DATA_DIR}/screening_data.csv')
mol_dict = pd.read_csv(f'{DATA_DIR}/eval_chemical_dict.csv')
with open(f'{DATA_DIR}/eval_chemical_feat.pkl', 'rb') as input_file:
  mol_feat = pickle.load(input_file)

# Define the core function
def retro_eval_core(model_type, train_df, test_df):
  count_fp = 2048 # length of fingerprints
  count_rdkit = 200 # length of descriptors
  outputs = []
  time_trackings = []
  # control models
  if model_type == 'ctrl':
    feat_start = time.time()
    X_train, y_train = ctrl_pair_feat(train_df, mol_dict=mol_dict, mol_feat=mol_feat, count_fp=count_fp, count_rdkit=count_rdkit)
    X_test, y_test = ctrl_pair_feat(test_df, mol_dict=mol_dict, mol_feat=mol_feat, count_fp=count_fp, count_rdkit=count_rdkit)
    feat_end = time.time()
    feat_duration = feat_end - feat_start
    for model in [XGBClassifier(n_estimators=500), RandomForestClassifier(n_estimators=500), MLPClassifier(), SVC(probability=True), GaussianProcessRegressor(), KNeighborsClassifier()]:
      output = model_build(model, model_type, X_train, y_train, X_test)
      outputs.append(output[:2])
      time_trackings.append([output[0], feat_duration, output[2], output[3], len(y_train), len(y_test)])

  elif model_type == 'kernel':
    feat_start = time.time()
    X_train, y_train, X_test, y_test = kernel_pair_feat(train_df, test_df, mol_dict=mol_dict, mol_feat=mol_feat, count_fp=count_fp, count_rdkit=count_rdkit)
    feat_end = time.time()
    feat_duration = feat_end - feat_start
    for model in [SVC(probability=True, kernel='precomputed'), GaussianProcessRegressor(kernel=PairwiseKernel(metric='precomputed')), KNeighborsClassifier(metric='precomputed')]:
      output = model_build(model, model_type, X_train, y_train, X_test)
      outputs.append(output[:2])
      time_trackings.append([output[0], feat_duration, output[2], output[3], len(y_train), len(y_test)])

  return outputs, time_trackings

# Define the evaluation pipeline
def retro_eval(mode, model_type, eval_df=eval_df):
  output_dfs = []
  time_tracking_dfs = []
  if mode in ['lodo', 'loeo']:
    col_name = 'drug_name' if mode == 'lodo' else 'excp_name'
    unique_mol = list(set(eval_df[col_name]))
    for mol in tqdm(unique_mol, desc=mode):
      train_df = pd.concat([prior_df, eval_df])
      # remove pairs that include the tested molecule
      train_df = train_df[train_df[col_name] != mol].copy()
      test_df = eval_df[eval_df[col_name] == mol].copy()
      print(len(train_df), len(test_df))
      outputs, time_trackings = retro_eval_core(model_type, train_df, test_df)
      for output in outputs:
        test_df.insert(loc=len(test_df.columns), column=output[0], value=output[1])
      output_dfs.append(test_df)
      time_tracking_dfs.append(time_trackings)

  elif mode == 'lopo':
    unique_pair = list(set([tuple((eval_df['drug_name'][i], eval_df['excp_name'][i])) for i in range(len(eval_df))]))
    for pair in tqdm(unique_pair, desc=mode):
      train_df = pd.concat([prior_df, eval_df])
      # remove pairs that include the tested molecule
      train_df = train_df[(train_df['drug_name'] != pair[0]) & (train_df['excp_name'] != pair[1])].copy()
      test_df = eval_df[(eval_df['drug_name'] == pair[0]) & (eval_df['excp_name'] == pair[1])].copy()
      print(len(train_df), len(test_df))
      outputs, time_trackings = retro_eval_core(model_type, train_df, test_df)
      for output in outputs:
        test_df.insert(loc=len(test_df.columns), column=output[0], value=output[1])
      output_dfs.append(test_df)
      time_tracking_dfs.append(time_trackings)

  else:
    raise KeyError("Please select from 'lodo', 'loeo' and 'lopo'. Double check your spelling.")

  output_df = pd.concat(output_dfs)
  output_df.insert(loc=0, column='eval_mode', value=[mode]*len(output_df))
  output_df.reset_index(inplace=True, drop=True)

  flat_time_tracking_dfs = [val for sublist in time_tracking_dfs for val in sublist]
  time_tracking_df = pd.DataFrame(data=flat_time_tracking_dfs, columns=['model_name', 'feat_duration_s', 'train_duration_s', 'predict_duration_s', 'train_count', 'predict_count'])
  time_tracking_df.insert(loc=0, value=[mode]*len(time_tracking_df), column='eval_mode')

  return output_df, time_tracking_df

# Summarize and export results
for model_type in ['kernel', 'ctrl']:
    for mode in ['lodo', 'loeo', 'lopo']:
        output_df, time_tracking_df = retro_eval(mode, model_type)
        export_to_csv(output_df, RESULTS_DIR, '{}_{}_result.csv'.format(mode, model_type))
        export_to_csv(time_tracking_df, RESULTS_DIR, '{}_{}_time.csv'.format(mode, model_type))

