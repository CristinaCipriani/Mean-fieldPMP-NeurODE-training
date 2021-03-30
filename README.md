# A Measure Theoretical Approach to the Mean-field Maximum Principle for Training NeurODE

This repository is the official implementation of[A Measure Theoretical Approach to the Mean-field Maximum Principle for Training NeurODE](https://arxiv.org/). 

>📋  Our theoretical approach leads to a new training algorithm for NeurODE which consists of a shooting method to solve the optimaliy conditions, namely a forward equation, a backward one and then an equation for the update of the control parameter, i.e. the weights of the layers of the network.

## Requirements

This implementation makes use of standard Python packages such as numpy, scipy, sklearn or matplotlib.

## Training

The algorithm can be trained by running this command:

```train
python train.py --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate the model , run:

```eval
python eval.py 
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Results


| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Citing

>📋 To cite our paper, use the following:
