# TuNaAI: a hybrid kernel machine to design tunable nanoparticles for drug delivery
This study combines *kernel machine design*, *lab automation*, and *experimental characterization techniques* to develop an **Tu**nable **Na**noparticle platform guided by **AI** (**TuNaAI**).

- Introduced the concept of **turning drug-excipient nanoparticles by adjusting stoichiometry** during synthesis.
- Developed **an automated, high-throughput data generation workflow**.
- Constructed **a bespoke kernel machine** (figure below) to guide the design of tunable nanoparticles.
- Enabled **encapsulation of previously inaccessible drugs** by rational increase of excipient.
- Computationally guided the reduction of excipient to **prepare potent and safer nanoparticles**.

  
![image](https://github.com/user-attachments/assets/99b0d86e-e8f9-4780-94b1-715728c855e3)

This work was presented at The 22nd International Nanomedicine and Drug Delivery Symposium ([NanoDDS2024](https://pharmacy.ufl.edu/2024/09/16/emerging-field-of-nanomedicine-takes-center-stage-as-uf-hosts-nanodds-symposium/)), and received a Trainee Travel Award.

## Dependency
Supervised machine learning runs in Python 3.9 using algorithms from [scikit-learn](https://scikit-learn.org/stable/) and [XGBoost](https://xgboost.readthedocs.io/en/stable/). [e3fp](https://github.com/keiserlab/e3fp) enables efficient calculation of tanimoto similarity. [tqdm](https://github.com/tqdm/tqdm) is a useful tool to visually track your job progress. A fresh conda environment can be set up using

```
conda create -n tuna python=3.9 pandas
conda activate tuna
conda install scikit-learn
conda install -c conda-forge py-xgboost
conda install -c conda-forge e3fp
conda install tqdm
```
Alternatively, users could implement the analysis on cloud-based platforms with pre-configured Python environment, e.g. Google Colab, and required packages can be installed using

```
!pip install xgboost
!pip install e3fp
```


## Descriptions of folders and files
### data
The available data sources include:
* Experimental data of DLS measurement
* Pre-calculated molecule features

### code
* This folder includes core functions that underlie the analysis pipeline and executable examples for users to run.
