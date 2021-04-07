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
from modules.training_1D import MFOC
from modules.analysis_1D import plot_loss_fct
from modules.analysis_1D import loss_comparison_Lambda, loss_comparison_dt

parser = argparse.ArgumentParser(description='Description of all the parameters below')
parser.add_argument("--dt", default=0.1, help="This is time-discretization dt")
parser.add_argument("--mu_0",
                    choices=["bigaussian", "gaussian"],
                    required=True, type=str,
                    help="This decides if the initial distirbution mu_0 is a bimodal or unimodal gaussian")
parser.add_argument("--Lambda", default=0.1, help="This is regularization parameter lambda")
parser.add_argument("--iterations", default=10, help="This is the number of outer iterations (of the shooting method)")

args = parser.parse_args()
dt = args.dt
mu_0 = args.mu_0
Lambda = args.Lambda
num_iterations = args.iterations


# Other parameters
N_points = 100
d = 1
T = 1
Nt = int(round(T/float(dt)))
print("dt is %s, hence the networks has %s layers" %(dt, Nt))
xmin = -7
xmax = 7
grid_points = 14

#Initial distribution
R = 0.2
if mu_0 == "bigaussian":
    center_left = -1
    center_right = 1
    mid_point = 0
    y_left = -2
    y_right = 2
else:
    center_left = 0
    center_right = 0
    mid_point = 0
    y_left = -1
    y_right = 1

#Initial guess of theta
theta = np.ones((Nt-1,d))

#Activation function
def F(x, theta):
    return np.tanh(theta*x)

print("END")
