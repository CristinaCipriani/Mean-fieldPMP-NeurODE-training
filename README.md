# A Measure Theoretical Approach to the Mean-field Maximum Principle for Training NeurODE

This repository is the official implementation of [A Measure Theoretical Approach to the Mean-field Maximum Principle for Training NeurODE](https://arxiv.org/). 

>📋  Our theoretical approach leads to a new training algorithm for NeurODE which consists of a shooting method to solve the optimaliy conditions, namely a forward equation, a backward one and then an equation for the update of the control parameter, i.e. the weights of the layers of the network.

## Requirements

This implementation makes use of standard Python packages such as numpy, scipy, sklearn or matplotlib.

## Training

The algorithm can be trained by running this command:

```train
python monodimensional.py --mu_0 bigaussian -- bias True --lambda 0.1 --dt 0.1 --iterations 8
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Results

The code automatically save two images in .png format, one is the one below that represents the movement of the particles, and the other one contains the evolution of the control parameter in time.

![alt text](https://github.com/CristinaCipriani/Mean-fieldPMP-NeurODE-training/blob/main/bimodal_evolution.PNG?raw=true)
![alt text](https://github.com/CristinaCipriani/Mean-fieldPMP-NeurODE-training/blob/main/evolution_theta_from_zeros.png?raw=true)

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Citing

>📋 To cite the paper, use the following:
