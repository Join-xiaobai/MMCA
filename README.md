# MMCA: Multi-Modal Co-Attention Model for Drug-Target Interaction Prediction
MMCA is a deep learning framework based on the **Multi-Modal Co-Attention** mechanism for accurate Drug-Target Interaction (DTI) prediction. It also supports anti-COVID-19 drug discovery and screening.

---

## Repository Overview
This repository contains the full implementation of the MMCA model, datasets, and running scripts. All code is open-sourced under the **MIT License**.

GitHub: https://github.com/Join-xiaobai/MMCA

## Requirements
- Python 3.9+
- PyTorch 1.13+
- CPU / GPU supported
- Dependencies: pandas, torch, torch_geometric, etc.

## File Structure
```
├── MMCA.py            # Main training script
├── test.py            # Model testing and evaluation
├── crossTest.py       # 5-fold cross-validation
├── datas.zip          # Experimental datasets (COVID-19 & drug–target interaction)
├── COVID-19.zip       # COVID-19 related data, code and results
├── COVID-19_drug_protein.csv       # prediction datas
├── data/
│   ├── drug_feature.csv      # Drug feature file
│   ├── protein_feature.csv   # Protein feature file
│   └── drug_protein.csv      # Drug-target interaction data
├── outputs/            # Training logs and output metrics
├── bestModel/          # Best saved model checkpoint
└── README.md
```

## Quick Start
1. Unzip the dataset
```
unzip datas.zip
```
2. Train the model
```
python MMCA.py
```
3. Test the model (update the model path first)
```
python test.py
```
4. Run 5-fold cross-validation
```
python crossTest.py
```

## Features
- Drug representation: Molecular graph + Mol2vec embedding
- Protein encoding: Anc2vec sequence embedding + handcrafted physicochemical features
- Multi-modal co-attention for drug–target feature fusion
- Output metrics: AUC, AUPR, F1-score, precision, recall
- Full reproducibility of the paper’s experimental results

## Hyperparameter Tuning
You can adjust the following parameters according to your hardware:
- batch_size
- learning_rate
- num_neighbors
- training epochs, hidden dimensions, etc.
