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


