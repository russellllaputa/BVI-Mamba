# BVI-Mamba

## Dataset Preparation

### 1. Download the dataset

Download the video dataset BVI-RLV from [here](https://dx.doi.org/10.21227/mzny-8c77).

Data Structure
```
.
└── BVI-RLV dataset
    ├── input
    │   ├── S01
    │   │   ├── low_light_10
    │   │   └── low_light_20
    │   ├── S02
    │   │   ├── low_light_10
    │   │   └── low_light_20
    │   └── ...
    └── gt
        ├── S01
        │   ├── normal_light_10
        │   └── normal_light_20
        ├── S02
        │   ├── normal_light_10
        │   └── normal_light_20
        └── ...
```

### 2. Modify the dataset path

Modify the dataset path in the config file `BVIMamba.yml`. 

### Train and test on other dataset  
Please modify `data_loader.py`  to train and test on other datasets.

## Train
```
python train.py
```

## Test
```
python test_fullresolution.py
```