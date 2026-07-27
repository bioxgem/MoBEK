# Integrating Drug-like Moieties and Binding Site Evolution for Kinase Inhibitor Prediction Using Ensemble Learning Models

This repository contains the primary datasets, trained Random Forest model, feature-generation code, and analysis notebooks associated with our kinase inhibitor prediction study.

![Overview of the kinase inhibitor prediction framework](overview.png)

**Overview of the kinase inhibitor prediction framework.**

## Project Structure

```text
.
├── README.md
├── overview.png
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── validation.csv
├── models/
│   └── random_forest_model.joblib
├── notebooks/
│   ├── 01_protein_feature_generation.ipynb
│   ├── 02_merge_compound_features.ipynb
│   ├── 03_random_forest_example.ipynb
│   └── 04_train_and_evaluate_model.ipynb
├── src/
│   └── feature_generate.py
└── resources/
    ├── protein/
    │   └── KLIFS_85_feature_list.txt
    └── moieties/
        ├── pubchem/
        ├── inhouse/
        └── rings_in_drugs/
```

## Requirements

- Python
- NumPy
- pandas
- scikit-learn
- RDKit
- Matplotlib
- Seaborn
- joblib
- SHAP
- Jupyter Notebook

Some molecular-moiety features rely on external structure-matching tools that are not redistributed in this first release.

## Data

| File | Purpose | Records |
| --- | --- | ---: |
| `data/train.csv` | Model training set | 147,583 |
| `data/test.csv` | Held-out test set | 36,911 |
| `data/validation.csv` | Independent validation set | 57,383 |

Each dataset contains 2,751 columns, including compound, kinase, label, molecular-feature, and binding-site-feature fields. The CSV files are configured for Git Large File Storage (Git LFS) through `.gitattributes`.

External-validation data is not included in this first release.

## Model

`models/random_forest_model.joblib` contains the trained Random Forest classifier used in this study. The corresponding training notebook uses 1,000 estimators with square-root feature sampling and no maximum-depth restriction.

The model file is configured for Git LFS through `.gitattributes`.

## Code and Notebooks

- `src/feature_generate.py` generates compound features from SMILES input and reads molecular definitions from `resources/moieties/`.
- `notebooks/01_protein_feature_generation.ipynb` records the binding-site feature-generation procedure.
- `notebooks/02_merge_compound_features.ipynb` merges compound-feature representations.
- `notebooks/03_random_forest_example.ipynb` demonstrates Random Forest training and evaluation.
- `notebooks/04_train_and_evaluate_model.ipynb` contains the main model-training and evaluation workflow.

## Usage

After obtaining the repository with Git LFS enabled, open the notebooks with Jupyter:

```bash
cd notebooks
jupyter notebook
```

To inspect the trained model in Python:

```python
import joblib

model = joblib.load("models/random_forest_model.joblib")
```

Adjust the model path if Python is started from a different working directory.

## Limitations

This first release preserves the principal datasets, trained model, and research workflow, but it is not intended as a complete end-to-end reproduction package.

- External-validation data is not included pending confirmation.
- The protein-feature notebook retains a historical `Filter_BBA.csv` input dependency that is not included in this release.
- External feature-matching executables are not redistributed.
