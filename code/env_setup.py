# === Imports ===
import pickle
import numpy as np
from sklearn.svm import SVC
from utils import *

# === Load Training Data ===
TRAIN_PKL_PATH = './TuNa-AI/data/train_info.pkl'
with open(TRAIN_PKL_PATH, 'rb') as infile:
    X_train, y_train = pickle.load(infile)

# === Core Prediction Function ===
def make_prediction(
    input_info,
    count_fp=2048,
    count_rdkit=200,
    X_train=X_train,
    y_train=y_train,
    l1=10,
    l2=10
):
    """
    Predict the outcome for a new system using a kernel SVC model.

    Parameters:
        input_info: tuple -> (drug_smiles, excp_smiles, mixing_ratio)
        count_fp: int, default=2048 -> fingerprint feature count
        count_rdkit: int, default=200 -> RDKit feature count
        X_train: np.ndarray -> training descriptors
        y_train: np.ndarray -> training labels
        l1: float -> RBF kernel parameter for descriptors
        l2: float -> RBF kernel parameter for ratio
    Returns:
        Prediction result from model_build
    """
    # Unpack input info
    drug_smiles, excp_smiles, mixing_ratio = input_info

    # Featurize new system
    drug_feat  = describe_mol(drug_smiles)
    excp_feat  = describe_mol(excp_smiles)
    X_test = np.concatenate([drug_feat, excp_feat, [mixing_ratio]]).reshape(1, -1)

    # Feature indices
    fp_index      = np.concatenate((np.arange(count_fp), np.arange(count_fp+count_rdkit, 2*count_fp+count_rdkit)))
    rdkit_index   = np.concatenate((np.arange(count_fp, count_fp+count_rdkit), np.arange(2*count_fp+count_rdkit, 2*count_fp+2*count_rdkit)))
    ratio_index   = np.array([-1])

    # Log2 ratio transformations
    train_ratio   = np.log2(X_train[:, ratio_index])
    test_ratio    = np.log2(X_test[:, ratio_index])

    # Kernel matrices for train and test
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

    # Combine kernels for chemistry and final kernel
    chemistry_train = matrices_operation(train_matrix[:2], option='addition')
    train_kernel = matrices_operation([chemistry_train, train_matrix[-1]], option='multiplication')

    chemistry_test = matrices_operation(test_matrix[:2], option='addition')
    test_kernel = matrices_operation([chemistry_test, test_matrix[-1]], option='multiplication')

    # Build model and predict
    model = SVC(probability=True, kernel='precomputed')
    return model_build(model, 'kernel', train_kernel, y_train, test_kernel)