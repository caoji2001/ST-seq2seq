# ST-seq2seq

## Overview

This repository contains an **unofficial** PyTorch implementation of the ["Cell-Level Trajectory Prediction Using Time-embedded Encoder-Decoder Network"](https://dl.acm.org/doi/10.1145/3615894.3628503) method, as part of the [HuMob Challenge 2023](https://connection.mit.edu/humob-challenge-2023).

## Setup

```bash
pip install -r requirements.txt
```

## Run

1. Prepare data

Download the data from [here](https://zenodo.org/records/10142719) and place it in the `data` directory.

2. Train

```bash
python train_task1.py

python train_task2.py
```

3. Predict

```bash
python val_task1.py

python val_task2.py
```
