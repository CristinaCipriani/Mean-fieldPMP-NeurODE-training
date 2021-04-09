# A Measure Theoretical Approach to the Mean-field Maximum Principle for Training NeurODE

This repository is the official implementation of [A Measure Theoretical Approach to the Mean-field Maximum Principle for Training NeurODE](https://arxiv.org/). 

>📋  Our theoretical approach leads to a training algorithm for NeurODE which consists of a shooting method to solve the optimaliy conditions, namely a forward equation, a backward one and then an equation for the update of the control parameter, i.e. the weights of the layers of the network.

## Requirements

This implementation makes use of standard Python packages such as numpy, scipy, sklearn and matplotlib.

## Training

In the monodimensional case, the algorithm can be trained by running this command:

```train
python monodimensional.py --mu_0 bigaussian -- bias True --lambda 0.1 --dt 0.1 --iterations 8
```
While in the bidimensional case, the command to run is the following:

```train
python bidimensional.py --mu_0 bigaussian -- bias True --lambda 0.1 --dt 0.1 --iterations 8
```

>📋  Entries: In both cases, the required parameter that the user need to choose is 'mu_0' which indicates if the initial distribution of particles is a bimodal gaussian (indicated with the term 'bigaussian') or a unimodal distribution (chosen by typing 'gaussian'). The centers can't be chosen by the user, but can be modified in the code. Moreover, the parameter 'bias' indicates if there's a bias term in the activation functions of the network. This is set by default to 'False', i.e. no bias. The other parameters that the user has the possibiliy to choose are 'dt', 'lambda' and 'iterations'. In Section 5.1 of the paper, it's explained how interesting is to play around with these parameters and see how they influence the final result.  

## Results

Both functions output at each step of every iteration the resolution of the forward and backward equation, and also the one of the parameter update. Automaticallay, plots of these functions are generated, but are not saved. While, at the end of the final iteration, two plots are saved in the current directory in .png format. One is the one below that represents the movement of the particles, and the other one contains the evolution of the control parameter in time.

First Output             |  Second Output
:-------------------------:|:-------------------------:
![](https://github.com/CristinaCipriani/Mean-fieldPMP-NeurODE-training/blob/main/bimodal_evolution.PNG)  |  ![](https://github.com/CristinaCipriani/Mean-fieldPMP-NeurODE-training/blob/main/bimodal_evolution.PNG)

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Citing

>📋 To cite the paper, use the following:
