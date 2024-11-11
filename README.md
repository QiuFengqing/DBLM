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

### MCM Dataset
The MCM dataset is publicly available, and we have preprocessed this dataset for use in our experiments.

### BSS Dataset
The large-scale BBS dataset for **15 minutes** supply-demanding prediction (797 stations, 10,207,268 trips, and over 3 years).

### SLD Dataset
The SLD dataset contains FHV trip records in Manhattan, divided into 67 zones by zip codes, from January to April 2018, aggregated every **5 minutes**.

### Additional Explanation
Our approach has broader potential for solving similar problems in urban traffic management. For example, vehicle inflows and outflows at various stations can be interpreted as order and supply volumes in the supplier allocation context. 
- If a station’s outflow exceeds its inflow, it indicates a high-demand area requiring additional vehicles. 
- Our model can provide valuable recommendations for vehicle allocation, helping to prevent shortages.

In the **SLD** and **BBS** datasets:
- **Inflow** corresponds to **order volume**.
- **Outflow** corresponds to **supply volume**.

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

