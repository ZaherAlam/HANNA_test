# HANNA-from-Scratch (VLE Modeling)

This repository implements a HANNA-style thermodynamically constrained neural network model to predict activity coefficients and VLE behavior.

## 📂 Repository Structure


hanna_from_scratch/
├── data/
│   ├── features.csv                  # Input features
│   └── targets.csv                   # Target values
├── models/
│   └── HANNA.py                      # Main model architecture
├── utils/
│   ├── scalers.py                    # Custom scaling utilities
│   ├── thermodynamics.py             # g^E, ln_gamma calculations
│   ├── embeddings.py
│   └── data_loader.py
├── train.py                          # Training loop
├── predict.py                        # Prediction script
├── config.yaml                       # Model/config parameters
├── evaluate.py
├── requirements.txt                  # Required packages
└── HANNAenvironment.yml              # Conda environment (from HANNA)
