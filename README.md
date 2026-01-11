# TuNa-AI: A Hybrid Kernel Machine to Design Tunable Nanoparticles for Drug Delivery
[This study](https://pubs.acs.org/doi/10.1021/acsnano.5c09066) combines *kernel machine design*, *lab automation*, and *experimental characterization techniques* to develop an **Tu**nable **Na**noparticle platform guided by **AI** (**TuNa-AI**).

- Introduced the concept of **tuning drug-excipient nanoparticles by adjusting stoichiometry** during synthesis.
- Developed **an automated, high-throughput data generation workflow**.
- Constructed **a bespoke kernel machine** (figure below) to guide the design of tunable nanoparticles.
- Enabled **encapsulation of previously inaccessible drugs** by rational increase of excipient.
- Computationally guided the reduction of excipient to **prepare potent and safer nanoparticles**.

![kernel](https://github.com/user-attachments/assets/3124451e-cb61-4c2f-8c56-ad99a9fc8741)


## Dependency
Supervised machine learning runs using algorithms from [scikit-learn](https://scikit-learn.org/stable/), [XGBoost](https://xgboost.readthedocs.io/en/stable/), [Chemprop](https://github.com/chemprop/chemprop) (V 1.5.1) and [GraphGPS](https://github.com/rampasek/GraphGPS). The [e3fp](https://github.com/keiserlab/e3fp) package facilitates the efficient calculation of Tanimoto similarity. [RDKit](https://www.rdkit.org/) and [DescriptaStorus](https://github.com/bp-kelley/descriptastorus) are chemoinformatics libraries designed for molecular featurization. Additionally, [tqdm](https://github.com/tqdm/tqdm) provides a convenient way to visually monitor job progress.

## Descriptions of folders and files
### data
The available data sources include:
* Historical drug-excipient nanoparticle data
* High-throughput screening data with various drug/excipient molar ratios
* Structure information of investigated chemicals

### code
This folder includes core functions that underlie the analysis pipeline and executable examples for users to run:
* Retrospective evaluation of machine learning and deep learning
* Prospective prediction of new compounds and pairs
* Real-time nanoparticle prediction through a [webserver](https://github.com/RekerLab/TuNa-AI/blob/main/code/TuNaOnline.md)

## License
The copyrights of the software are owned by Duke University. As such, two licenses for this software are offered:
1. An open-source license under the GPLv2 license for non-commercial academic use.
2. A custom license with Duke University, for commercial use or uses without the GPLv2 license restrictions.

## Citation
If you find this work or code useful, please cite our [ACS Nano paper](https://pubs.acs.org/doi/10.1021/acsnano.5c09066).
