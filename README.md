# TuNa-AI: a hybrid kernel machine to design tunable nanoparticles for drug delivery
This study combines *kernel machine design*, *lab automation*, and *experimental characterization techniques* to develop an **Tu**nable **Na**noparticle platform guided by **AI** (**TuNa-AI**).

- Introduced the concept of **tuning drug-excipient nanoparticles by adjusting stoichiometry** during synthesis.
- Developed **an automated, high-throughput data generation workflow**.
- Constructed **a bespoke kernel machine** (figure below) to guide the design of tunable nanoparticles.
- Enabled **encapsulation of previously inaccessible drugs** by rational increase of excipient.
- Computationally guided the reduction of excipient to **prepare potent and safer nanoparticles**.

![kernel](https://github.com/user-attachments/assets/3124451e-cb61-4c2f-8c56-ad99a9fc8741)


## Dependency
Supervised machine learning runs in Python 3.9 using algorithms from [scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.readthedocs.io/en/stable/) and [Chemprop](https://github.com/chemprop/chemprop). The [e3fp](https://github.com/keiserlab/e3fp) package facilitates the efficient calculation of Tanimoto similarity. [RDKit](https://www.rdkit.org/) and [DescriptaStorus](https://github.com/bp-kelley/descriptastorus) are chemoinformatics libraries designed for molecular featurization. Additionally, [tqdm](https://github.com/tqdm/tqdm) provides a convenient way to visually monitor job progress. 

## Descriptions of folders and files
### data
The available data sources include:
* Experimental data of DLS measurement
* Pre-calculated molecule features

### code
* This folder includes core functions that underlie the analysis pipeline and executable examples for users to run.
