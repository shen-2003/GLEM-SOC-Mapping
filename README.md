# GLEM-SOC-Mapping

## A global–local ensemble learning framework with adaptive spatial weighting for soil organic carbon mapping

This repository provides the source code and supporting data for the implementation of the global–local ensemble learning framework with adaptive spatial weighting developed for soil organic carbon (SOC) mapping.

## 1. Overview

Soil organic carbon (SOC) exhibits complex spatial heterogeneity resulting from the combined effects of global environmental trends and local spatial processes. To address this challenge, this study develops a global–local ensemble learning framework with adaptive spatial weighting that integrates complementary global and local modeling strategies.

The proposed framework combines three base learners:

- Mixed-Effects Model (MixedLM) for capturing global linear relationships;
- Support Vector Regression (SVR) for capturing global nonlinear relationships;
- Multiscale Geographically Weighted Regression (MGWR) for capturing local spatially varying relationships.

The predictions of the three base learners are subsequently integrated using adaptive spatial weights. The weighting module uses model prediction errors and spatial relationships to determine the relative contributions of the base learners, allowing the ensemble model to adapt to spatially heterogeneous modeling performance.

## 2. Framework

The proposed global–local ensemble learning framework consists of the following main components:

1. Data preprocessing and spatial partitioning;
2. Training of the three base learners;
3. Generation of base-model predictions;
4. Calculation of prediction residuals;
5. Construction of adaptive spatial weights;
6. Integration of global and local model predictions;
7. Evaluation of SOC prediction performance.

The three base learners provide complementary representations of SOC–environment relationships. MixedLM captures globally consistent linear patterns, SVR represents global nonlinear relationships, and MGWR accounts for locally varying spatial relationships. The adaptive spatial weighting strategy then determines their relative contributions across geographic space.

## 3. Repository Structure

```text
GLEM-SOC-Mapping/
└── GLEM_SOC/
    ├── data/
    │   └── Supporting datasets
    ├── GLEM_soc.py
    ├── MGWR_model_soc.py
    ├── mixedlm_model_soc.py
    └── SVM_model_soc.py
```

### Main Scripts

| File | Description |
|---|---|
| `GLEM_soc.py` | Implementation of the global–local ensemble learning framework |
| `MGWR_model_soc.py` | Multiscale Geographically Weighted Regression model |
| `mixedlm_model_soc.py` | Mixed-Effects Model |
| `SVM_model_soc.py` | Support Vector Regression model |
| `data/` | Supporting datasets used in the study |

## 4. Software Requirements

The source code was developed and tested using Python 3.9.

The main Python packages required by the framework include:

- NumPy
- pandas
- scikit-learn
- statsmodels
- mgwr
- PyTorch
- Optuna

## 5. Installation

A Python 3.9 environment is recommended.

The main required packages can be installed using:

```bash
pip install numpy pandas scikit-learn statsmodels mgwr torch optuna
```

## 6. Data Availability

The supporting datasets used for model development and evaluation are provided in the `GLEM_SOC/data/` directory.

The data included in this repository are publicly available and are provided to support the reproducibility of the analyses presented in the associated research article.

## 7. Usage

The modeling workflow is implemented using the scripts provided in the `GLEM_SOC/` directory.

The three base learners are implemented in:

```text
SVM_model_soc.py
MGWR_model_soc.py
mixedlm_model_soc.py
```

The outputs of the three base learners are subsequently used by:

```text
GLEM_soc.py
```

to perform global–local ensemble learning with adaptive spatial weighting and generate the final SOC predictions.

Before running the scripts, users should ensure that the input data paths and output directories are correctly configured according to their local computing environment.

## 8. Reproducibility

The repository provides the source code and supporting datasets required to reproduce the main modeling procedures described in the associated research article.

The code implements the main modeling workflow, including the construction of the three base learners and the proposed global–local ensemble learning framework with adaptive spatial weighting.

## 9. Research Article

This repository is associated with the following research article:

**A global–local ensemble learning framework with adaptive spatial weighting for soil organic carbon mapping**

The final bibliographic information will be updated after publication.

## 10. Citation

If you use the source code or datasets provided in this repository, please cite the associated research article:

> A global–local ensemble learning framework with adaptive spatial weighting for soil organic carbon mapping.

## 11. Software and Data Availability

The source code and supporting datasets used in this study are openly available in this repository:

**Source code:**  
https://github.com/shen-2003/GLEM-SOC-Mapping

**Programming language:**  
Python 3.9

The repository contains the implementation of the three base learners and the proposed global–local ensemble learning framework with adaptive spatial weighting, together with the supporting datasets required for model development and evaluation.

## 12. License

This project is licensed under the MIT License.
