## Expected Folder Structure
```
data/
│── datasets/                          # Directory for multiple datasets
│   ├── dataset_1/                      # First dataset (e.g., CCD)
│   │   ├── raw/                         # Raw data (videos, annotations, etc.)
│   │   ├── processed/                    # Processed features
│   │   ├── metadata.csv                  # Metadata about the dataset
│   │   ├── config.yaml                    # Dataset-specific config
│   │   ├── stats/                         # Dataset statistics
│   │   ├── README.md                      # Documentation for this dataset
│   │
│   ├── dataset_2/                      # Second dataset (e.g., DAD)
│   │   ├── raw/
│   │   ├── processed/
│   │   ├── metadata.csv
│   │   ├── config.yaml
│   │   ├── stats/
│   │   ├── README.md
│   │
│   ├── dataset_3/                      # Third dataset (e.g., Another custom dataset)
│   │   ├── raw/
│   │   ├── processed/
│   │   ├── metadata.csv
│   │   ├── config.yaml
│   │   ├── stats/
│   │   ├── README.md
│
│── preprocessing/                      # Scripts for dataset preprocessing
│   ├── extract_features.py              # Extracts features for all datasets
│   ├── convert_npy_to_tensors.py        # Converts `.npy` features to tensors
│   ├── normalize_features.py            # Normalization of features
│   ├── preprocess_dataset.py            # Main script (handles multiple datasets)
│
│── utils.py                             # Helper functions (loading data, parsing configs)
│── dataset_loader.py                    # Generalized DataLoader for multiple datasets
│── dataset_config.yaml                   # Global config to define active dataset
│── __init__.py                           # To make `data/` a package
```
