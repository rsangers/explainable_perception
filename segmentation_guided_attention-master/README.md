# Segmentation guided attention
Pytorch implementation for the paper: Reconciling explainability and performance in neural networks by means of semantic segmentation-guided feature attention:An application to urban space perception (unpublished).

## Requirements

Python >= 3.6.5 (only tested on that one)

For more check `requirements.txt`

## Setup and preprocessing

First install dependencies

`pip install -r requirements.txt`

Get the dataset and put all the images in a single `placepulse/` folder in the root directory. Also put the complete `votes.csv` file in the root directory.
After that run the preprocessing scripts.

```bash
python image_crop.py
python place_pulse_clean.py
python placepulse_split.py
```

## Training
Now you can start training:

```bash
python train.py
```
For information on the different parameters run:

```bash
python train.py -h
```

