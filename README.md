# Explainability of Deep Learning models for Urban Space Perception

Python code for the paper: Explainability of Deep Learning models for Urban Space Perception.

## Setup and preprocessing

This repository has been tested on Windows with Python 3.8.

The dependencies can be installed using:
`pip install -r requirements.txt`

This project uses the Place Pulse 2.0 dataset from Dubey et al. [1]. To use this dataset, place all images in a folder called `placepulse/` in the root directory, as well as the `votes.csv` file containing all voting data. The code of this repository builds upon the work of A.C. Vidal [2].

To preprocess the dataset, it is advised to run the following scripts:
```bash
python image_crop.py
python place_pulse_clean.py
python placepulse_split.py
```

## Training
Training of the models can be started by running the `train.py` file. Parameters such as the model to use and the attribute to train on can be specified by passing command line arguments. For more information, please run:
```bash
python train.py -h
```

## Analysis and explainability methods
Pretrained models can be found in the `models/` directory. These can be evaluated using the `run_cam.py` script. Parameters such as the explainability method to use can be specified at the end of this file.

## References
[1] Dubey, Abhimanyu, et al. "Deep learning the city: Quantifying urban perception at a global scale." European conference on computer vision. Springer, Cham, 2016.

[2] Vidal, Andrés Cádiz. Deep Neural Network Models with Explainable Components for Urban Space Perception. Diss. Pontificia Universidad Catolica de Chile (Chile), 2021.
