# -*- coding: utf-8 -*-
# Import dependencies
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel
from e3fp.fingerprint.metrics.array_metrics import tanimoto
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from descriptastorus.descriptors import rdNormalizedDescriptors

# Molecule featurization
def describe_mol(smiles):
  mol = Chem.MolFromSmiles(smiles)
  # morgan fingerprints
  fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048) # radius=4, bits=2048
  fp_list = []
  fp_list.extend(fp.ToBitString())
  fp_expl = []
  fp_expl = [int(x) for x in fp_list]
  # normalized physicochemical (rdkit) descriptors
  generator = rdNormalizedDescriptors.RDKit2DNormalized()
  descriptor = generator.process(smiles)[1:]
  # concatenation
  representation = fp_expl + descriptor
  return representation

# Retrieve pre-calculated features if available
def name2feat(df, mol_dict, mol_feat, label=True):
  drug_name = df['drug_name'].to_list()
  excp_name = df['excp_name'].to_list()
  ratio_list = df['excp_drug_ratio'].to_list()

  mol_dict = {name: smiles for name, smiles in zip(mol_dict['name'].to_list(), mol_dict['smiles'].to_list())}
  drug_smiles = [mol_dict[name] for name in drug_name]
  excp_smiles = [mol_dict[name] for name in excp_name]

  feat_array = np.array([mol_feat[drug_smiles[i]] + mol_feat[excp_smiles[i]] + [ratio_list[i]] for i in range(len(df))])

  if label:
    return feat_array, np.array(df['class'])
  else:
    return feat_array

# Matrix operation
def matrices_operation(matrix_list, option='multiplication'):
  if not matrix_list:
    raise ValueError("The input list of matrices is empty.")
  # Initialize the result matrix with the identity matrix of the same shape as the first matrix in the list
  if option == 'multiplication':
    result = np.ones(matrix_list[0].shape[1], dtype=int)
  elif option == 'addition':
    result = np.zeros(matrix_list[0].shape[1], dtype=int)
  else:
    raise KeyError("Please select from 'multiplication' and 'addition'. Double check your spelling.")

  # Multiply each matrix element-wise with the result, or element-wise addition
  for matrix in matrix_list:
    if option == 'multiplication':
      result = np.multiply(result, matrix)
    elif option == 'addition':
      result = result + matrix
    else:
      raise KeyError("Please select from 'multiplication' and 'addition'. Double check your spelling.")
  return result

# Concatenate all features as inputs for standard ML models
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

# Hybrid kernel featurization
def kernel_pair_feat(train_df, test_df, mol_dict, mol_feat, count_fp=2048, count_rdkit=200, retro=True, test_mol_dict=None, test_mol_feat=None):
  X_train, y_train = name2feat(train_df, mol_dict, mol_feat)
  if retro: # retrospective evaluation
    X_test, y_test = name2feat(test_df, mol_dict, mol_feat)
  else: # prospective prediction
    X_test = name2feat(test_df, test_mol_dict, test_mol_feat, label=retro)

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
  
  if retro: # retrospective evaluation
    return train_matrix, y_train, test_matrix, y_test
  else: # prospective prediction
    return train_matrix, y_train, test_matrix    

# Convert similarity to distance matrix
def sim2dis(similarity_matrix, reference_point):
  distance_matrix = reference_point - similarity_matrix
  return distance_matrix

# Build the model
def model_build(model, model_type, X_train, y_train, X_test):
  if model_type not in ['ctrl', 'kernel']:
    raise KeyError("Please select from 'ctrl' and 'kernel'. Double check your spelling.")
  else:
    model_name = f'{model_type}_{str(type(model).__name__)}'
  print(f'Developing {model_name}...')
  train_start = time.time()
  if model_type == 'kernel' and 'KNeighborsClassifier' in model_name:
    X_train, X_test = sim2dis(X_train, 3), sim2dis(X_test, 3)
  model.fit(X_train, y_train)
  train_end = time.time()
  proba = 1 - model.predict_proba(X_test)[:, 0] if 'GaussianProcessRegressor' not in model_name else model.predict(X_test)
  predict_end = time.time()

  return [model_name, proba, train_end-train_start, predict_end-train_end]

# Export results
def export_to_csv(df, folder, filename):
  # Ensure the results directory exists
  os.makedirs(folder, exist_ok=True)
  # Export the dataFrame to a CSV file
  df.to_csv(os.path.join(folder, filename), index=False)
