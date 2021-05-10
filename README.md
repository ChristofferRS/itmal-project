# ITMAL project

## Structure

```
.
├── data
│   ├── 0_dB_pump.zip
│   └── pump
│       ├── id_00
│       │   ├── abnormal
│       │   └── normal
│       ├── id_02
│       │   ├── abnormal
│       │   └── normal
│       ├── id_04
│       │   ├── abnormal
│       │   └── normal
│       └── id_06
│           ├── abnormal
│           └── normal
├── makefile
├── nn.py
├── README.md
├── requirements.txt
└── tools
    ├── __init__.py
    ├── prepro.py
    └── show.py

```

- Tools dir for import tools and feature extracting tools
- datadir for the datafiles
- makefile for downloading audio data and cleaning and so on
- requirements.txt for managing requirements. Good to use with virtualenv

## Model 1

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
normalization (Normalization (None, 1598, 20, 1)       3
_________________________________________________________________
conv2d (Conv2D)              (None, 1596, 18, 60)      600
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 1594, 16, 60)      32460
_________________________________________________________________
flatten (Flatten)            (None, 1530240)           0
_________________________________________________________________
dense (Dense)                (None, 60)                91814460
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 122
=================================================================
Total params: 91,847,645
Trainable params: 91,847,642
Non-trainable params: 3
_________________________________________________________________
```

## Model 2

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
resizing (Resizing)          (None, 32, 32, 1)         0
_________________________________________________________________
normalization (Normalization (None, 32, 32, 1)         3
_________________________________________________________________
conv2d (Conv2D)              (None, 30, 30, 60)        600
_________________________________________________________________
flatten (Flatten)            (None, 54000)             0
_________________________________________________________________
dense (Dense)                (None, 60)                3240060
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 122
=================================================================
Total params: 3,240,785
Trainable params: 3,240,782
Non-trainable params: 3
_________________________________________________________________
```
