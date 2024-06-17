# BVI-Mamba

## Dataset

### BVI-RLV

### Data Preparation
Please generate and replace ```data/video_input.txt``` and ```data/video_test.txt``` with your own paths of low-light training and testing images before continue.

## Train
### Option 1: recompile DCNv2 PyTorch C++ extensions from [BasicSR](https://github.com/XPixelGroup/BasicSR) during installation
```
BASICSR_EXT=True python setup.py develop
```
```
python train.py
```

### Option 2: run from scratch and load the DCNv2 PyTorch C++ extensions just-in-time (JIT)
``` 
BASICSR_JIT=True python train.py
```

## Test