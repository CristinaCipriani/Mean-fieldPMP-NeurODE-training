import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
%matplotlib inline
import time
from IPython import display
from scipy import stats
from scipy import interpolate
from sklearn.neighbors import KernelDensity
from modules.training import MFOC
from modules.analysis import plot_loss_fct
from modules.analysis import loss_comparison_Lambda, loss_comparison_dt

N_points = 100
d = 1
T = 1
dt = 0.1 #0.1 #0.05
Nt = int(round(T/float(dt)))
print("dt is %s, hence the networks has %s layers" %(dt, Nt))

# Initial distribution
R = 0.2
mu_0 = "bigaussian"
center_left = -1
center_right = 1
mid_point = 0
y_left = -2
y_right = 2

# Mesh
xmin = -7
xmax = 7
grid_points = 141 #71 #141

#Initial guess of theta
theta = -1*np.ones((Nt-1,d))
#theta = np.random.rand(Nt-1,d)

#Activation function
def F(x, theta):
    return np.tanh(theta*x)

#Regularization parameter
Lambda = 0.1

#Number of iterations
num_iterations = 8 #31

theta, theta_trace, mu = MFOC(N_points, d, T, dt, R, mu_0, center_left, center_right, y_left, y_right, xmin, xmax, grid_points, theta, F, mid_point, Lambda, num_iterations)
