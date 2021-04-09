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

parser = argparse.ArgumentParser(description='Description of all the parameters below')
parser.add_argument("--mu_0",
                    choices=["bigaussian", "gaussian"],
                    required=True, type=str,
                    help="This decides if the initial distirbution mu_0 is a bimodal or unimodal gaussian")
parser.add_argument("--bias", default = False,
                                        help="This decides if the activation function contains a bias or not")
parser.add_argument("--dt", default=0.1, help="This is time-discretization dt")
parser.add_argument("--Lambda", default=0.1, help="This is regularization parameter lambda")
parser.add_argument("--iterations", default=10, help="This is the number of outer iterations (of the shooting method)")

args = parser.parse_args()
mu_0 = args.mu_0
bias = args.bias
dt = args.dt
lbd = args.Lambda
num_iterations = args.iterations

# Setting the right format
dt = np.float(dt)
lbd = np.float(lbd)
num_iterations = np.int(num_iterations)

#Other parameters
N_points = 100
d = 2
T = 1
dt = 0.1
Nt = int(round(T/float(dt)))
print("dt is %s, hence the networks has %s layers" %(dt, Nt))
xmin = -3
xmax = 3
grid_points = 61

# Initial distribution
R = 0.2
if mu_0 == "bigaussian":
    center_left = np.array([-1, -1])
    center_right = np.array([1, 1])
    mid_point = 0
    y_left = np.array([-2, -2])
    y_right = np.array([2, 2])
else:
    center_left = np.array([0, 0])
    center_right = np.array([0, 0])
    mid_point = 0
    y_left = np.array([-1, -1])
    y_right = np.array([1, 1])

#Activation functions
def F_nobias(x, theta):
    return np.tanh(theta @ x)

def F_bias(x, theta):
    return np.tanh(theta[:,:d] @ x + theta[:,d])

if bias == False:
    # Setting the parameters needed for the case without bias
    theta = 5*np.ones((Nt-1,d,d))
    F = F_nobias
    Lambda = lbd*np.ones((d,d))

    # Running the algorithm
    theta, theta_trace = MFOC_nobias(N_points, d, T, dt, R, mu_0, center_left, center_right, y_left, y_right, xmin, xmax, grid_points, theta, F, mid_point, Lambda, num_iterations)

    # Plotting the evolution of theta and saving it in the current directory (MISSING)

else:
    # Setting the parameters needed for the case with bias
    theta = np.ones((Nt-1,d,d+1))
    F = F_bias
    Lambda = lbd*np.ones((d,d+1))
    Lambda[:, d] = 0.1*np.ones(d)

    # Running the algorithm
    theta, theta_trace = MFOC_bias(N_points, d, T, dt, R, mu_0, center_left, center_right, y_left, y_right, xmin, xmax, grid_points, theta, F, mid_point, Lambda, num_iterations)

    # Plotting the evolution of theta and saving it in the current directory (MISSING)

print("End of training, two images have been plotted in the current directory")
