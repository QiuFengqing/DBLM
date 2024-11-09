This is a Pytorch implementation of DBLM: Time Series Supplier Allocation via Deep Black-Litterman Model

# Time Series Supplier Allocation via Deep Black-Litterman Model

More details of the paper and dataset will be released after it is published.


# The Code

## Requirements

Following is the suggested way to install the dependencies:

    conda install --file DBLM.yml

Note that ``pytorch >=1.10``.

## Folder Structure

```tex
└── code-and-data
    ├── MCM                     # The MCM-TSSO Dataset
    ├── BBS                     # The traffic management domain Dataset
    ├── SLD                     # The large-scale TTSO Dataset
    ├── methods                 # The core source code of our model DBLM
    │   |──  _init_.py          # Initialization file for models
    │   |──  DBLM.py            # Including the model in DBLM    
    ├── dataloader              # Contains the SOS dataloader 
    ├── mian.py                 # This is the main file
    ├── evaluation.py           # Evaluation matrics
    ├── loss_function.py        # The mask rank loss function
    ├── DBLM.yml                # The python environment needed for DBLM
    └── README.md               # This document
```

## Datasets

The MCM dataset is public and we have preprocessed this dataset. 

## Configuration

```tex
nhid = 150              # The hidden unit of Graph Encoder and Predictor
risk_k = 2              # The power valuen of risk, i.e., \kappa
BLM_tau = 3             # The power of tau, i.e., \tau
BLM_delta = 0.6.        # The coefficient balancing risk and profit, i.e., \delta
```


##  Train and Test

Simply run  `"main.py"` with your own dataset name (e.g.,  MCM) and you can start training and testing your model.

We provide more options for you for further study:

- ```tex
  --mask_ratio          # The ratio for masking order and supply data
  ```

# Reference

```
```

