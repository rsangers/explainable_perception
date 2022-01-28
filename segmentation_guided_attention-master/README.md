# Explainability of Deep Learning models for UrbanSpace Perception

Python code for the paper: Explainability of Deep Learning models for Urban Space Perception

## Setup and preprocessing

This repository has been tested on Windows with Python 3.8.

The dependencies can be installed using
`pip install -r requirements.txt`

This project uses the Place Pulse 2.0 dataset from Dubey et al. [1]. To use this dataset, place all images in a folder called `placepulse/` in the root directory, as well as the `votes.csv` file containing all voting data.

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

## References
[1] Dubey, Abhimanyu, et al. "Deep learning the city: Quantifying urban perception at a global scale." European conference on computer vision. Springer, Cham, 2016.
