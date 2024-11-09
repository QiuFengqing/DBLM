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
    ├── BBS                     # The 15-minute Large-scale TTSO Dataset for Traffic Management
    ├── SLD                     # The 5-minute Large-scale TTSO Dataset for Traffic Management
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

The BBS dataset includes trip records from ride-sharing companies in Chicago’s 77 zones every 15 minutes.

The SLD dataset includes trip records from ride-sharing companies in Chicago’s 77 zones every 5 minutes.

More explanation: We also recognize the broader potential of our approach for similar problems. For instance, in urban traffic management, vehicle inflows and outflows at various stations can be likened to order and supply volumes in the supplier allocation context. If a station’s outflow exceeds its inflow, this indicates a high-demand area where additional vehicles are needed. Our model could provide valuable recommendations for vehicle allocation, helping to prevent shortages. In the dataset SLD and BBS, inflow->order, outflow->supply.

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

