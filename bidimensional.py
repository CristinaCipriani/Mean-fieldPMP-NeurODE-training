import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
#%matplotlib inline
import time
from IPython import display
from scipy import stats
from scipy import interpolate
from sklearn.neighbors import KernelDensity
from modules.training_nobias_2D import MFOC as MFOC_nobias
from modules.training_bias_2D import MFOC as MFOC_bias
