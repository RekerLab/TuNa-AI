# -*- coding: utf-8 -*-
### Loading utility tools
import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

### supervised machine learning models and evaluation
## importing machine learning libraries and core packages
import xgboost
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.metrics.pairwise import rbf_kernel
from e3fp.fingerprint.metrics.array_metrics import tanimoto


## defining utility functions
# Convert a readable dataframe of chemical name pairs into precomputed features
def name2feat(df, mol_dict, mol_feat):
  drug_name = df['drug_name'].to_list()
  excp_name = df['excp_name'].to_list()
  ratio_list = df['excp_drug_ratio'].to_list()

  drug_smile = [mol_dict[mol_dict['name'] == drug_name[i]]['smile'].to_list()[0] for i in range(len(df))]
  excp_smile = [mol_dict[mol_dict['name'] == excp_name[i]]['smile'].to_list()[0] for i in range(len(df))]

  feat_array = np.array([mol_feat[drug_smile[i]] + mol_feat[excp_smile[i]] + [ratio_list[i]] for i in range(len(df))])

  return feat_array, np.array(df['class'])

def matrices_operation(matrix_list, option='multiplication'):
  if not matrix_list:
    raise ValueError("The input list of matrices is empty.")
  # Initialize the result matrix with the identity matrix of the same shape as the first matrix in the list
  if option == 'multiplication':
    result = np.ones(matrix_list[0].shape[1], dtype=int)
  elif option == 'addition':
    result = np.zeros(matrix_list[0].shape[1], dtype=int)

  # Multiply each matrix element-wise with the result, or element-wise addition
  for matrix in matrix_list:
    if option == 'multiplication':
      result = np.multiply(result, matrix)
    elif option == 'addition':
      result = result + matrix
  return result

def ctrl_pair_feat(df, mol_dict, mol_feat, count_fp=2048, count_rdkit=200):
  X, y = name2feat(df=df, mol_dict=mol_dict, mol_feat=mol_feat)
  drug_feat = X[:, :count_fp+count_rdkit]
  excp_feat = X[:, count_fp+count_rdkit:2*(count_fp+count_rdkit)]
  ratio = X[:, -1]
  # morgan-rdkit, molar ratio as an individual column, concatenation
  X = [np.concatenate((drug_feat[i], excp_feat[i], np.array([ratio[i]]))) for i in tqdm(range(len(df)))]
  # summary
  X = pd.DataFrame(X)
  return X, y

def kernel_pair_feat(train_df, test_df, mol_dict, mol_feat, count_fp=2048, count_rdkit=200):
  X_train, y_train = name2feat(train_df, mol_dict, mol_feat)
  X_test, y_test = name2feat(test_df, mol_dict, mol_feat)
  fp_index = np.concatenate((np.arange(count_fp), np.arange(count_fp+count_rdkit, 2*count_fp+count_rdkit)))
  rdkit_index = np.concatenate((np.arange(count_fp, count_fp+count_rdkit), np.arange(2*count_fp+count_rdkit, 2*count_fp+2*count_rdkit)))
  ratio_index = np.array([-1])
  train_ratio = np.array([np.log2(i) for i in X_train[:, ratio_index]])
  test_ratio = np.array([np.log2(i) for i in X_test[:, ratio_index]])
  l1, l2 = 10, 10

  train_matrix = [
  tanimoto(X_train[:, fp_index]), 
  rbf_kernel(X_train[:, rdkit_index], gamma=1/(2*l1*l1)), 
  rbf_kernel(train_ratio, gamma=1/(2*l2*l2))
  ]
  test_matrix = [
  tanimoto(X_test[:, fp_index], X_train[:, fp_index]), 
  rbf_kernel(X_test[:, rdkit_index], X_train[:, rdkit_index], gamma=1/(2*l1*l1)), 
  rbf_kernel(test_ratio, train_ratio, gamma=1/(2*l2*l2))
  ]

  chemistry_train = matrices_operation(train_matrix[:2], option='addition')
  train_matrix = matrices_operation([chemistry_train, train_matrix[-1]], option='multiplication')

  chemistry_test = matrices_operation(test_matrix[:2], option='addition')
  test_matrix = matrices_operation([chemistry_test, test_matrix[-1]], option='multiplication')

  return train_matrix, y_train, test_matrix, y_test

def sim2dis(similarity_matrix, reference_point):
  # convert similarity matrix to distance/dissimilarity matrix
  distance_matrix = reference_point - similarity_matrix
  return distance_matrix

def export_to_csv(df, folder, filename):
  # Create the folder if it doesn't exist
  if not os.path.exists(folder):
    os.makedirs(folder)

  # Export the DataFrame to a CSV file
  df.to_csv(os.path.join(folder, filename), index=False)

# load libraries
#from functions import *
import pickle

# load prior knowledge and evaluation data
prior_df = pd.read_csv('../data/prior_data.csv')
eval_df = pd.read_csv('../data/pair_with_ratio.csv')
# load precomputed mol feature
input = open('../data/chemical_feat.pkl', 'rb')
mol_feat = pickle.load(input)
input.close()
# load chemical dictionary mapping names to smiles
mol_dict = pd.read_csv('../data/chemical_dict.csv')

def retro_eval_core(model_type, train_df, test_df):
  count_fp = 2048 # length of fingerprints
  count_rdkit = 200 # length of descriptors
  outputs = []
  time_trackings = []
  if model_type == 'ctrl':
    feat_start = time.time()
    X_train, y_train = ctrl_pair_feat(train_df, mol_dict=mol_dict, mol_feat=mol_feat, count_fp=count_fp, count_rdkit=count_rdkit)
    X_test, y_test = ctrl_pair_feat(test_df, mol_dict=mol_dict, mol_feat=mol_feat, count_fp=count_fp, count_rdkit=count_rdkit)
    feat_end = time.time()
    for model in [XGBClassifier(n_estimators=500), RandomForestClassifier(n_estimators=500), MLPClassifier(), SVC(probability=True), GaussianProcessRegressor(), KNeighborsClassifier()]:
      model_name = 'ctrl_' + str(type(model).__name__)
      train_start = time.time()
      model.fit(X_train, y_train)
      train_end = time.time()
      proba = [1-i for i in [model.predict_proba(X_test)[:, 0]]][0] if 'GaussianProcessRegressor' not in model_name else model.predict(X_test)
      predict_end = time.time()
      outputs.append([model_name, proba])
      # track time use
      feat_length = feat_end - feat_start
      train_length = train_end - train_start
      predict_length = predict_end - train_end
      time_trackings.append([model_name, feat_length, train_length, predict_length, len(y_train), len(y_test)])


  elif model_type == 'kernel':
    feat_start = time.time()
    X_train, y_train, X_test, y_test = kernel_pair_feat(train_df, test_df, mol_dict=mol_dict, mol_feat=mol_feat, count_fp=count_fp, count_rdkit=count_rdkit)
    feat_end = time.time()
    for model in [SVC(probability=True, kernel='precomputed'), GaussianProcessRegressor(kernel=PairwiseKernel(metric='precomputed')), KNeighborsClassifier(metric='precomputed')]:
      model_name = 'kernel_' + str(type(model).__name__)
      train_start = time.time()
      if 'KNeighborsClassifier' in model_name:
        X_train = sim2dis(X_train, 3)
        X_test = sim2dis(X_test, 3)
      model.fit(X_train, y_train)
      train_end = time.time()
      proba = [1-i for i in [model.predict_proba(X_test)[:, 0]]][0] if 'GaussianProcessRegressor' not in model_name else model.predict(X_test)
      predict_end = time.time()
      outputs.append([model_name, proba])
      # track time use
      feat_length = feat_end - feat_start
      train_length = train_end - train_start
      predict_length = predict_end - train_end
      time_trackings.append([model_name, feat_length, train_length, predict_length, len(y_train), len(y_test)])

  return outputs, time_trackings

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
  output_df = pd.concat(output_dfs)
  output_df.insert(loc=0, column='eval_mode', value=[mode]*len(output_df))
  output_df.reset_index(inplace=True, drop=True)

  flat_time_tracking_dfs = [val for sublist in time_tracking_dfs for val in sublist]
  time_tracking_df = pd.DataFrame(data=flat_time_tracking_dfs, columns=['model_name', 'feat_length', 'train_length', 'predict_length', 'count_train', 'count_predict'])
  time_tracking_df.insert(loc=0, value=[mode]*len(time_tracking_df), column='eval_mode')

  return output_df, time_tracking_df

for model_type in ['kernel', 'ctrl']:
    for mode in ['lodo', 'loeo', 'lopo']:
        output_df, time_tracking_df = retro_eval(mode, model_type)
        export_to_csv(output_df, folder, '{}_{}_result.csv'.format(mode, model_type))
        export_to_csv(time_tracking_df, folder, '{}_{}_time.csv'.format(mode, model_type))
