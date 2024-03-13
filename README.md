# Differentially Private Federated Learning


## Overview
This repository contains code for the implementation and experimentation of various neural network architectures trained on medical image classification tasks using PyTorch. The primary focus is on Federated Learning (FL) and Differentially Private Federated Learning (DPFL) settings, aiming to address privacy concerns in healthcare data. The project utilizes the MedMNIST dataset, specifically PathMNIST, BloodMNIST, and OrganAMNIST, although other datasets can be easily added and experimented with.

Paper: [Research Gate](https://www.researchgate.net/publication/377524548_A_Differentially_Private_Federated_Learning_Application_in_Privacy-Preserving_Medical_Imaging)

## Requirements
- Python 3.x
- PyTorch
- Other dependencies specified in `Pipfile`

## Usage
### Configuration
- The project includes a `config.yaml` file which contains general settings and configurations.
- Modify the `global_config` section for general settings such as seed, device, multiprocess settings (`is_mp`), differential privacy settings (`is_dp`), and whether to save confusion matrices (`save_cm`).
- The `fed_config` section contains settings specific to Federated Learning, including fraction of clients (`C`), number of clients (`K`), number of rounds (`R`), local epochs (`E`), batch size (`B`), criterion, and optimizer. The number of clients (`K`) accepts a list, allowing experiments with different numbers of clients (e.g., `[2, 8, 32]`).

### Training
- To train models in Federated Learning settings, run `main.py` after ensuring `is_dp` is set to `Fasle`.
- For Differentially Private Federated Learning settings, ensure `is_dp` is set to `True` in the configuration file and run the `main.py` script.

### Adding Architectures/Datasets
- Additional architectures can be easily added to the project.
- To experiment with other datasets, simply add the dataset name to the `datasets_names` value in `config.yaml`.


## Citation
If you find this repository useful in your research, please consider citing:


FARES, Mohamad & SERTBAŞ, Ahmet. (2024). A Differentially Private Federated Learning Application in Privacy-Preserving Medical Imaging. https://doi.org/10.21203/rs.3.rs-3873379/v2.
