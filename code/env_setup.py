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
    try:
        drug_feat  = describe_mol(drug_smiles)
    except:
        print('Invalid drug SMILES!')
        return

    try:
        excp_feat  = describe_mol(excp_smiles)
    except:
        print('Invalid excipient SMILES!')
        return

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
    proba = model_build(model, 'kernel', train_kernel, y_train, test_kernel, verbose=0)[1][0]
    output = f'{np.round(proba*100, 2)}%'
    return output

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# Helper for styled text widgets (no internal description)
def styled_text_input(placeholder, value=''):
    return widgets.Text(
        value=value,
        placeholder=placeholder,
        description='',  # no duplicate label
        layout=widgets.Layout(height='60px')
    )

# Title
title = widgets.HTML("<h2>Free that Tuna! 🧪 Predict your nanoparticle formation with tunability </h2>")

# Input widgets
drug_smiles_input = styled_text_input('Enter drug SMILES')
excp_smiles_input = styled_text_input('Enter excipient SMILES')
mixing_ratio_input = styled_text_input('Enter positive number', '1.0')

submit_button = widgets.Button(description='Predict', layout=widgets.Layout(height='60px'))
output_area = widgets.Output()

# Validation functions
def is_valid_string(s):
    return isinstance(s, str) and s.strip() != ''

def is_valid_number(s):
    try:
        val = float(s)
        return val > 0
    except:
        return False

# Button click logic
def on_button_clicked(b):
    with output_area:
        clear_output()
        drug_smiles = drug_smiles_input.value
        excp_smiles = excp_smiles_input.value
        mixing_ratio_str = mixing_ratio_input.value

        # Validate SMILES
        if not is_valid_string(drug_smiles):
            display(HTML("<span style='font-size:16px; color:red'>Error: Drug SMILES must be a non-empty string.</span>"))
            return
        if not is_valid_string(excp_smiles):
            display(HTML("<span style='font-size:16px; color:red'>Error: Excipient SMILES must be a non-empty string.</span>"))
            return

        # Validate mixing ratio
        if not is_valid_string(mixing_ratio_str):
            display(HTML("<span style='font-size:16px; color:red'>Error: Mixing ratio must be a string representing a number.</span>"))
            return
        if not is_valid_number(mixing_ratio_str):
            display(HTML("<span style='font-size:16px; color:red'>Error: Mixing ratio must be a positive number.</span>"))
            return

        # Convert and predict
        mixing_ratio = float(mixing_ratio_str)
        try:
            result = make_prediction((drug_smiles.strip(), excp_smiles.strip(), mixing_ratio))
            display(HTML(f"<span style='font-size:16px;'>Model prediction: {result}</span>"))
        except Exception as e:
            display(HTML(f"<span style='font-size:16px; color:red'>Error: {e}</span>"))

submit_button.on_click(on_button_clicked)

# Assemble clean layout with large bold labels
dashboard = widgets.VBox([
    widgets.HTML("<div style='font-size:16px; font-weight:bold'>Drug SMILES:</div>"),
    drug_smiles_input,
    widgets.HTML("<div style='font-size:16px; font-weight:bold'>Excipient SMILES:</div>"),
    excp_smiles_input,
    widgets.HTML("<div style='font-size:16px; font-weight:bold'>Mixing Excipient-to-Drug Ratio:</div>"),
    mixing_ratio_input,
    submit_button,
    output_area
])