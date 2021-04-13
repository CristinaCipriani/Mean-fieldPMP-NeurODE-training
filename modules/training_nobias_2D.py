import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
from IPython import display
from scipy import stats
from scipy import interpolate
from scipy import optimize
from sklearn.neighbors import KernelDensity

def trunc_gaussian(shape, R):
    X = np.random.normal(size = shape)
    for i in range(shape[0]):
        while np.linalg.norm(X[i, :]) > R:
            X[i, :] = np.random.normal(size = shape[1])
    return X

def initial_distribution(y_left, y_right, N, mu_0, d, R, mid_point, center_left, center_right):
    # Creation of the initial distribution
    Y_left = np.tile(y_left, (N,1))
    Y_right = np.tile(y_right, (N,1))

    if mu_0 == "gaussian" :
        X_all = trunc_gaussian(shape=(2*N,d), R=R) + mid_point
        Y_all = np.ones((2*N,d))
        for i in range(0,2*N):
            if X_all[i,0] > mid_point:
                Y_all[i] = y_right
            else:
                Y_all[i] = y_left
        Z_all = np.concatenate([X_all, Y_all], axis=1)
    else:
        X_left = trunc_gaussian(shape=(N,d), R=R) + center_left
        Z_left = np.concatenate([X_left, Y_left], axis=1)
        X_right = trunc_gaussian(shape=(N,d), R=R) + center_right
        Z_right = np.concatenate([X_right, Y_right], axis=1)
        X_all = np.concatenate([X_left,X_right], axis=0)
        Z_all = np.concatenate([Z_left,Z_right], axis=0)

    return Z_all

def I(x1, x2, y1, y2):
    x = [x1, x2]
    y = [y1, y2]
    return np.linalg.norm(np.array(x)-np.array(y))**2

def F1(nt,x1,x2,theta_in):
    th11 = theta_in[nt,0,0]
    th12 = theta_in[nt,0,1]
    return np.tanh(th11*x1 + th12*x2)

def F2(nt,x1,x2,theta_in):
    th21 = theta_in[nt,1,0]
    th22 = theta_in[nt,1,1]
    return np.tanh(th21*x1 + th22*x2)

def der_theta_F1(nt, x1, x2, theta_in):
    th11 = theta_in[nt,0,0]
    th12 = theta_in[nt,0,1]
    const = 1-F1(nt,x1, x2,theta_in)**2
    mat = np.zeros((2,2))
    mat[0,0] = const * x1
    mat[0,1] = const * x2
    return mat

def der_theta_F2(nt, x1, x2, theta_in):
    th21 = theta_in[nt,1,0]
    th22 = theta_in[nt,1,1]
    const = 1-F2(nt,x1, x2,theta_in)**2
    mat = np.zeros((2,2))
    mat[1,0] = const * x1
    mat[1,1] = const * x2
    return mat

def move_forward(dt, Nt, N_particles, Z, F, theta, mid_point, y_left, y_right):
    # Resolution of the forward equation
    d = theta.shape[1]

    Z_trace = np.zeros(shape=(Nt,2*N_particles,2*d))
    Z_trace[0,:,:] = Z

    g = np.zeros(shape=(Nt,2*N_particles,d))
    g[0,:,0] = y_right[0] * np.sign(Z[:,0])
    g[0,:,1] = y_right[1] * np.sign(Z[:,0])

    for n in range(0, Nt-1):
        for i in range(2*N_particles):
            Z_trace[n+1,i,:d] = Z_trace[n,i,:d] + dt * F(Z_trace[n,i,:d], theta[n, :,:])
            Z_trace[n+1,i,d:] = Z_trace[n,i,d:]

        g[n+1,:,0] = y_right[0] * np.sign(Z_trace[n+1,:,0])
        g[n+1,:,1] = y_right[1] * np.sign(Z_trace[n+1,:,0])

    return Z_trace, g

def move_backward(x1, x2, y1, y2, xmin, xmax, dx, Nx, dt, Nt, I, F1, F2, theta, diff):
    # Resolution of the backward equation
    Nx1 = Nx
    Nx2 = Nx
    Ny1 = Nx
    Ny2 = Nx
    dx1 = dx
    dx2 = dx
    dy1 = dx
    dy2 = dx

    u = np.zeros((Nx1+2, Nx2+2, Ny1+2, Ny2+2)) # unknown u at new time level
    u_n = np.zeros((Nx1+2, Nx2+2, Ny1+2, Ny2+2)) # u at the previous time level
    u_all = np.zeros((Nt, Nx1, Nx2, Ny1, Ny2))

    #Auxiliary variables for the ghost points
    x1_b = np.linspace(xmin-dx1, xmax+dx1, Nx1+2)
    x2_b = np.linspace(xmin-dx2, xmax+dx2, Nx2+2)
    y1_b = np.linspace(xmin-dy1, xmax+dy1, Ny1+2)
    y2_b = np.linspace(xmin-dy2, xmax+dy2, Ny2+2)

    # Load initial condition into u_n
    for i in range(0, Nx1+2):
        for j in range(0, Nx2+2):
            for l in range(0, Ny1+2):
                for m in range(0, Ny2+2):
                    u_n[i,j,l,m] = I(x1_b[i], x2_b[j], y1_b[l], y2_b[m])  #used when I is a function of x,y
    u_all[Nt-1,:,:,:,:] = u_n[1:-1,1:-1,1:-1,1:-1]

    for n in range(0, Nt-1):
        if diff == "fu":
            #Boundary points
            u[0,0,0:Ny1+2,0:Ny2+2] = (u_n[0,0,0:Ny1+2,0:Ny2+2]
                          - (-F1(Nt-2-n,x1_b[0],x2_b[0],theta))*dt/dx *
                            (-2*u_n[1,0,0:Ny1+2,0:Ny2+2]+3/2*u_n[0,0,0:Ny1+2,0:Ny2+2]+1/2*u_n[2,0,0:Ny1+2,0:Ny2+2])*0.5*(1+np.sign(-F1(Nt-2-n,x1_b[0],x2_b[0],theta)))
                          - (-F1(Nt-2-n,x1_b[0],x2_b[0],theta))*dt/dx *
                            (u_n[1,0,0:Ny1+2,0:Ny2+2] -u_n[0,0,0:Ny1+2,0:Ny2+2])*0.5*(1-np.sign(-F1(Nt-2-n,x1_b[1],x2_b[0],theta)))
                          - (-F2(Nt-2-n,x1_b[0],x2_b[0],theta))*dt/dx *
                            (-2*u_n[0,1,0:Ny1+2,0:Ny2+2]+3/2*u_n[0,0,0:Ny1+2,0:Ny2+2]+1/2*u_n[0,2,0:Ny1+2,0:Ny2+2])*0.5*(1+np.sign(-F2(Nt-2-n,x1_b[0],x2_b[0],theta)))
                          - (-F2(Nt-2-n,x1_b[0],x2_b[1],theta))*dt/dx *
                            (u_n[0,1,0:Ny1+2,0:Ny2+2]-u_n[0,0,0:Ny1+2,0:Ny2+2])*0.5*(1-np.sign(-F2(Nt-2-n,x1_b[0],x2_b[1],theta)))   #this part is correct
                         )
            u[0,Nx2+1,0:Ny1+2,0:Ny2+2] = (u_n[0,Nx2+1,0:Ny1+2,0:Ny2+2]
                          - (-F1(Nt-2-n,x1_b[0],x2_b[Nx2+1],theta))*dt/dx *
                            (-2*u_n[1,Nx2+1,0:Ny1+2,0:Ny2+2]+3/2*u_n[0,Nx2+1,0:Ny1+2,0:Ny2+2]+1/2*u_n[2,Nx2+1,0:Ny1+2,0:Ny2+2])*0.5*(1+np.sign(-F1(Nt-2-n,x1_b[0],x2_b[Nx2+1],theta)))
                          - (-F1(Nt-2-n,x1_b[1],x2_b[Nx2+1],theta))*dt/dx *
                            (u_n[1,Nx2+1,0:Ny1+2,0:Ny2+2] -u_n[0,Nx2+1,0:Ny1+2,0:Ny2+2])*0.5*(1-np.sign(-F1(Nt-2-n,x1_b[1],x2_b[Nx2+1],theta)))
                          - (-F2(Nt-2-n,x1_b[0],x2_b[Nx2+1],theta))*dt/dx *
                            (u_n[0,Nx2+1,0:Ny1+2,0:Ny2+2]-u_n[0,Nx2,0:Ny1+2,0:Ny2+2]) *0.5*(1+np.sign(-F2(Nt-2-n,x1_b[0],x2_b[Nx2],theta)))
                          - (-F2(Nt-2-n,x1_b[0],x2_b[Nx2+1],theta))*dt/dx *
                            (-3/2*u_n[0,Nx2+1,0:Ny1+2,0:Ny2+2]+2*u_n[0,Nx2,0:Ny1+2,0:Ny2+2]-1/2*u_n[0,Nx2-1,0:Ny1+2,0:Ny2+2]) *0.5*(1-np.sign(-F2(Nt-2-n,x1_b[0],x2_b[Nx2+1],theta)))
                              )                  #this part is correct
            u[Nx1+1,0,0:Ny1+2,0:Ny2+2] = (u_n[Nx1+1,0,0:Ny1+2,0:Ny2+2]
                          - (-F1(Nt-2-n,x1_b[Nx1+1],x2_b[0],theta))*dt/dx *
                            (u_n[Nx1+1,0,0:Ny1+2,0:Ny2+2]-u_n[Nx1,0,0:Ny1+2,0:Ny2+2]) *0.5*(1+np.sign(-F1(Nt-2-n,x1_b[Nx1],x2_b[0],theta)))
                          - (-F1(Nt-2-n,x1_b[Nx1],x2_b[0],theta))*dt/dx *
                            (-3/2*u_n[Nx1+1,0,0:Ny1+2,0:Ny2+2]+2*u_n[Nx1,0,0:Ny1+2,0:Ny2+2]-1/2*u_n[Nx1-1,0,0:Ny1+2,0:Ny2+2]) *0.5*(1-np.sign(-F1(Nt-2-n,x1_b[Nx1],x2_b[0],theta)))
                          - (-F2(Nt-2-n,x1_b[Nx1+1],x2_b[0],theta))*dt/dx *
                            (-2*u_n[Nx1+1,1,0:Ny1+2,0:Ny2+2]+3/2*u_n[Nx1+1,0,0:Ny1+2,0:Ny2+2]+1/2*u_n[Nx1+1,2,0:Ny1+2,0:Ny2+2])*0.5*(1+np.sign(-F2(Nt-2-n,x1_b[Nx1+1],x2_b[0],theta)))
                          - (-F2(Nt-2-n,x1_b[Nx1+1],x2_b[1],theta))*dt/dx *
                            (u_n[Nx1+1,1,0:Ny1+2,0:Ny2+2]-u_n[Nx1+1,0,0:Ny1+2,0:Ny2+2])*0.5*(1-np.sign(-F2(Nt-2-n,x1_b[Nx1+1],x2_b[1],theta)))   #this part is correct
                              )
            u[Nx1+1, Nx2+1,0:Ny1+2,0:Ny2+2] = (u_n[Nx1+1,Nx2+1,0:Ny1+2,0:Ny2+2]
                          - (-F1(Nt-2-n,x1_b[Nx1+1],x2_b[Nx2+1],theta))*dt/dx *
                            (u_n[Nx1+1,Nx2+1,0:Ny1+2,0:Ny2+2]-u_n[Nx1,Nx2+1,0:Ny1+2,0:Ny2+2]) *0.5*(1+np.sign(-F1(Nt-2-n,x1_b[Nx1],x2_b[Nx2+1],theta)))
                          - (-F1(Nt-2-n,x1_b[Nx1],x2_b[Nx2+1],theta))*dt/dx *
                            (-3/2*u_n[Nx1+1,Nx2+1,0:Ny1+2,0:Ny2+2]+2*u_n[Nx1,Nx2+1,0:Ny1+2,0:Ny2+2]-1/2*u_n[Nx1-1,Nx2+1,0:Ny1+2,0:Ny2+2]) *0.5*(1-np.sign(-F1(Nt-2-n,x1_b[Nx1],x2_b[Nx2+1],theta)))
                          - (-F2(Nt-2-n,x1_b[Nx1+1],x2_b[Nx2+1],theta))*dt/dx *
                            (u_n[Nx1+1,Nx2+1,0:Ny1+2,0:Ny2+2]-u_n[Nx1+1,Nx2,0:Ny1+2,0:Ny2+2]) *0.5*(1+np.sign(-F2(Nt-2-n,x1_b[Nx1+1],x2_b[Nx2+1],theta)))
                          - (-F2(Nt-2-n,x1_b[Nx1+1],x2_b[Nx2],theta))*dt/dx *
                            (-3/2*u_n[Nx1+1,Nx2+1,0:Ny1+2,0:Ny2+2]+2*u_n[Nx1+1,Nx2,0:Ny1+2,0:Ny2+2]-1/2*u_n[Nx1+1,Nx2-1,0:Ny1+2,0:Ny2+2]) *0.5*(1-np.sign(-F2(Nt-2-n,x1_b[Nx1+1],x2_b[Nx2],theta)))
                      )
            #Boundary lines
            for l in range(0,Ny1+2):
                    for m in range(0,Ny2+2):
                        u[0,1:Nx2+1,l,m] = (u_n[0,1:Nx2+1,l,m]
                                            - (-F1(Nt-2-n,x1_b[0],x2_b[1:Nx2+1],theta))*dt/dx1 *
                                              (-2*u_n[1,1:Nx2+1,l,m]+3/2*u_n[0,1:Nx2+1,l,m]+1/2*u_n[2,1:Nx2+1,l,m])
                                              *0.5*(1+np.sign(-F1(Nt-2-n,x1_b[0],x2_b[1:Nx2+1],theta)))
                                            - (-F1(Nt-2-n,x1_b[0],x2_b[1:Nx2+1],theta))*dt/dx1 *
                                              (-u_n[0,1:Nx2+1,l,m]+u_n[1,1:Nx2+1,l,m])
                                              *0.5*(1-np.sign(-F1(Nt-2-n,x1_b[0],x2_b[1:Nx2+1],theta)))
                                            - (-F2(Nt-2-n,x1_b[0],x2_b[1:Nx2+1],theta))*dt/dx2 *
                                              (-u_n[0,0:Nx2,l,m]+u_n[0,1:Nx2+1,l,m])
                                              *0.5*(1+np.sign(-F2(Nt-2-n,x1_b[0],x2_b[1:Nx2+1],theta)))
                                            - (-F2(n,x1_b[0],x2_b[0:Nx2],theta))*dt/dx2 *
                                              (-u_n[0,1:Nx2+1,l,m]+u_n[0,2:Nx2+2,l,m])
                                              *0.5*(1-np.sign(-F2(n,x1_b[0],x2_b[0:Nx2],theta)))   #ok
                                            )
                        u[1:Nx1+1,0,l,m] = (u_n[1:Nx1+1,0,l,m]
                                            - (-F1(Nt-2-n,x1_b[1:Nx1+1],x2_b[0],theta))*dt/dx1 *
                                              (u_n[1:Nx1+1,0,l,m]-u_n[0:Nx1,0,l,m])
                                              *0.5*(1+np.sign(-F1(Nt-2-n,x1_b[1:Nx1+1],x2_b[0],theta)))
                                            - (-F1(Nt-2-n,x1_b[0:Nx1],x2_b[0],theta))*dt/dx1 *
                                              (-u_n[1:Nx1+1,0,l,m]+u_n[2:Nx1+2,0,l,m])
                                              *0.5*(1-np.sign(-F1(Nt-2-n,x1_b[0:Nx1],x2_b[0],theta)))
                                            - (-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[0],theta))*dt/dx2 *
                                              (3/2*u_n[1:Nx1+1,0,l,m]-2*u_n[1:Nx1+1,1,l,m]+1/2*u_n[1:Nx1+1,2,l,m])
                                              *0.5*(1+np.sign(-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[0],theta)))
                                            - (-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[0],theta))*dt/dx2 *
                                              (u_n[1:Nx1+1,1,l,m]-u_n[1:Nx1+1,0,l,m])
                                              *0.5*(1-np.sign(-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[0],theta)))  #ok
                                            )
                        u[1:Nx1+1,Nx2+1,l,m] = (u_n[1:Nx1+1,Nx2+1,l,m]
                                               - (-F1(Nt-2-n,x1_b[1:Nx1+1],x2_b[Nx2+1],theta))*dt/dx1 *
                                                 (u_n[1:Nx1+1,Nx2+1,l,m]-u_n[0:Nx1,Nx2+1,l,m])
                                                 *0.5*(1+np.sign(-F1(Nt-2-n,x1_b[1:Nx1+1],x2_b[Nx2+1],theta)))
                                               - (-F1(Nt-2-n,x1_b[0:Nx1],x2_b[Nx2+1],theta))*dt/dx1 *
                                                 (u_n[2:Nx1+2,Nx2+1,l,m]-u_n[1:Nx1+1,Nx2+1,l,m])
                                                 *0.5*(1-np.sign(-F1(Nt-2-n,x1_b[0:Nx1],x2_b[Nx2+1],theta)))
                                               - (-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[Nx2+1],theta))*dt/dx2 *
                                                 (-u_n[1:Nx1+1,Nx2,l,m]+u_n[1:Nx1+1,Nx2+1,l,m])
                                                 *0.5*(1+np.sign(-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[Nx2+1],theta)))
                                               - (-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[Nx2],theta))*dt/dx2 *
                                                 (2*u_n[1:Nx1+1,Nx2,l,m]-3/2*u_n[1:Nx1+1,Nx2+1,l,m]-1/2*u_n[1:Nx1+1,Nx2-1,l,m])
                                                 *0.5*(1-np.sign(-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[Nx2],theta)))     #ok
                                            )
                        u[Nx+1,1:Nx2+1,l,m] = (u_n[Nx1+1,1:Nx2+1,l,m]
                                                - (-F1(Nt-2-n,x1_b[Nx1+1],x2_b[1:Nx2+1],theta))*dt/dx1 *
                                                  (u_n[Nx1+1,1:Nx2+1,l,m]-u_n[Nx1,1:Nx2+1,l,m])
                                                  *0.5*(1+np.sign(-F1(Nt-2-n,x1_b[Nx1+1],x2_b[1:Nx2+1],theta)))
                                                - (-F1(Nt-2-n,x1_b[Nx1],x2_b[1:Nx2+1],theta))*dt/dx1 *
                                                  (2*u_n[Nx1,1:Nx2+1,l,m]-3/2*u_n[Nx1+1,1:Nx2+1,l,m]-1/2*u_n[Nx1-1,1:Nx2+1,l,m])
                                                  *0.5*(1-np.sign(-F1(Nt-2-n,x1_b[Nx1],x2_b[1:Nx2+1],theta)))
                                                - (-F2(Nt-2-n,x1_b[Nx1+1],x2_b[1:Nx2+1],theta))*dt/dx2 *
                                                  (u_n[Nx1+1,1:Nx2+1,l,m]-u_n[Nx1+1,0:Nx2,l,m])
                                                  *0.5*(1+np.sign(-F2(Nt-2-n,x1_b[Nx1+1],x2_b[1:Nx2+1],theta)))
                                                - (-F2(Nt-2-n,x1_b[0],x2_b[0:Nx2],theta))*dt/dx2 *
                                                  (u_n[Nx1+1,2:Nx2+2,l,m]-u_n[Nx1+1,1:Nx2+1,l,m])
                                                  *0.5*(1-np.sign(-F2(Nt-2-n,x1_b[Nx1+1],x2_b[0:Nx2],theta)))  #ok
                                                )
            # Internal points
            u[1:Nx1+1,1:Nx2+1,1:Ny1+1,1:Ny2+1] = (u_n[1:Nx1+1,1:Nx2+1,1:Ny1+1,1:Ny2+1]
                                               - (-F1(Nt-2-n,x1_b[1:Nx1+1],x2_b[1:Nx2+1],theta))*dt/dx1 *
                                                 (u_n[1:Nx1+1,1:Nx2+1,1:Ny1+1,1:Ny2+1]-u_n[0:Nx1,1:Nx2+1,1:Ny1+1,1:Ny2+1])
                                                  *0.5*(1+np.sign(-F1(Nt-2-n,x1_b[1:Nx1+1],x2_b[1:Nx2+1],theta)))
                                               - (-F1(Nt-2-n,x1_b[0:Nx1],x2_b[1:Nx2+1],theta))*dt/dx1 *
                                                  (u_n[2:Nx1+2,1:Nx2+1,1:Ny1+1,1:Ny2+1]-u_n[1:Nx1+1,1:Nx2+1,1:Ny1+1,1:Ny2+1])
                                                  *0.5*(1-np.sign(-F1(Nt-2-n,x1_b[0:Nx1],x2_b[1:Nx2+1],theta)))
                                               - (-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[1:Nx2+1],theta))*dt/dx2 *
                                                 (-u_n[1:Nx1+1,0:Nx2,1:Ny1+1,1:Ny2+1]+u_n[1:Nx1+1,1:Nx2+1,1:Ny1+1,1:Ny2+1])
                                                  *0.5*(1+np.sign(-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[1:Nx2+1],theta)))
                                               - (-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[0:Nx2],theta))*dt/dx2 *
                                                 (u_n[1:Nx1+1,2:Nx2+2,1:Ny1+1,1:Ny2+1]-u_n[1:Nx1+1,1:Nx2+1,1:Ny1+1,1:Ny2+1])
                                                  *0.5*(1-np.sign(-F2(Nt-2-n,x1_b[1:Nx1+1],x2_b[0:Nx2],theta)))
                                                 )                                                               #ok

        u_all[Nt-2-n,:,:,:,:] = u[1:-1,1:-1,1:-1,1:-1]
        # Switch variables before next step
        u_n, u = u, u_n

    return u_all


def parameter_update(Z_trace, g, psi_pos, psi_neg, der_theta_F1, der_theta_F2, theta, Lambda, x1 , x2, dx , dt, Nx, Nt, mid_point):
    # Resolution of equation for the update of the parameter
    d = theta.shape[1]
    theta_new = np.zeros((Nt-1,d,d))
    count = 0
    N_particles = Z_trace.shape[1]

    for n in range(0, Nt-1):
        spl_neg = interpolate.RectBivariateSpline(x1, x2, psi_neg[n,:,:], kx=2, ky=2)
        spl_pos = interpolate.RectBivariateSpline(x1, x2, psi_pos[n,:,:], kx=2, ky=2)

        sol0 = optimize.brentq(root_function_00, -500, 500, args=(Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta, n))
        theta_new[n,0,0] = sol0

        sol1 = optimize.brentq(root_function_01, -500, 500, args=(Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta, n))
        theta_new[n,0,1] = sol1

        sol2 = optimize.brentq(root_function_10, -500, 500, args=(Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta, n))
        theta_new[n,1,0] = sol2

        sol3 = optimize.brentq(root_function_11, -500, 500, args=(Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta, n))
        theta_new[n,1,1] = sol3

        count += root_function(Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta_new, n)[4]

    # Calculating the loss function
    loss_fct = 0
    for i in range(0,N_particles):
        loss_fct += (1/N_particles) * np.linalg.norm(Z_trace[Nt-2,i,:2]-Z_trace[Nt-2,i,2:])**2
    loss_fct += (dt* (Lambda[0,0] * np.linalg.norm(theta[:,0,0])**2 + Lambda[0,1]*np.linalg.norm(theta[:,0,1])**2
                     +Lambda[1,0] * np.linalg.norm(theta[:,1,0])**2 + Lambda[1,1]*np.linalg.norm(theta[:,1,1])**2)
                     )

    return theta_new, count, loss_fct

def root_function(Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta, n):
    N_particles = Z_trace.shape[1]
    d = theta.shape[1]
    count = 0
    f = np.zeros((d,d))
    f_prime = np.zeros((d,d))

    for i in range(0,N_particles):
        x_i = Z_trace[n,i,:d]
        #x_i = np.random.multivariate_normal(Z_trace[n,i,:d],0.01*np.identity(d))
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] >= 0 :
            f += ( spl_pos(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_pos(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] < 0 :
            f += ( spl_neg(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_neg(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )
        if Z_trace[n,i,2] != g[n,i,0] or Z_trace[n,i,3] != g[n,i,1] :
            count += 1

    double = False
    if double == False:
        f0_tot = 2*Lambda[0,0]*theta[n,0,0] + f[0,0]/N_particles
        f1_tot = 2*Lambda[0,1]*theta[n,0,1] + f[0,1]/N_particles
        f2_tot = 2*Lambda[1,0]*theta[n,1,0] + f[1,0]/N_particles
        f3_tot = 2*Lambda[1,1]*theta[n,1,1] + f[1,1]/N_particles
    else:
        f0_tot = 2*Lambda[0,0]*theta[n,0,0] + 2 * f[0,0]/N_particles
        f1_tot = 2*Lambda[0,1]*theta[n,0,1] + 2 * f[0,1]/N_particles
        f2_tot = 2*Lambda[1,0]*theta[n,1,0] + 2 * f[1,0]/N_particles
        f3_tot = 2*Lambda[1,1]*theta[n,1,1] + 2 * f[1,1]/N_particles

    return f0_tot, f1_tot, f2_tot, f3_tot, count

def root_function_00(theta_00, Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta, n):
    theta[n,0,0] = theta_00
    N_particles = Z_trace.shape[1]
    d = theta.shape[1]
    f = np.zeros((d,d))

    for i in range(0,N_particles):
        x_i = Z_trace[n,i,:d]
        #x_i = np.random.multivariate_normal(Z_trace[n,i,:d],0.01*np.identity(d))
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] > 0 :
            f += ( spl_pos(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_pos(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] <= 0 :
            f += ( spl_neg(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_neg(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )

    double = False
    if double == False:
        f0_tot = 2*Lambda[0,0]*theta[n,0,0] + f[0,0]/N_particles
    else:
        f0_tot = 2*Lambda[0,0]*theta[n,0,0] + 2 * f[0,0]/N_particles

    return f0_tot

def root_function_01(theta_01, Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta, n):
    theta[n,0,1] = theta_01
    N_particles = Z_trace.shape[1]
    d = theta.shape[1]
    f = np.zeros((d,d))

    for i in range(0,N_particles):
        x_i = Z_trace[n,i,:d]
        #x_i = np.random.multivariate_normal(Z_trace[n,i,:d],0.01*np.identity(d))
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] > 0 :
            f += ( spl_pos(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_pos(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] <= 0 :
            f += ( spl_neg(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_neg(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )

    double = False
    if double == False:
        f1_tot = 2*Lambda[0,1]*theta[n,0,1] + f[0,1]/N_particles
    else:
        f1_tot = 2*Lambda[0,1]*theta[n,0,1] + 2 * f[0,1]/N_particles

    return f1_tot

def root_function_10(theta_10, Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta, n):
    theta[n,1,0] = theta_10
    N_particles = Z_trace.shape[1]
    d = theta.shape[1]
    f = np.zeros((d,d))

    for i in range(0,N_particles):
        x_i = Z_trace[n,i,:d]
        #x_i = np.random.multivariate_normal(Z_trace[n,i,:d],0.01*np.identity(d))
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] > 0 :
            f += ( spl_pos(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_pos(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] <= 0 :
            f += ( spl_neg(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_neg(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )

    double = False
    if double == False:
        f2_tot = 2*Lambda[1,0]*theta[n,1,0] + f[1,0]/N_particles
    else:
        f2_tot = 2*Lambda[1,0]*theta[n,1,0] + 2 * f[1,0]/N_particles

    return f2_tot

def root_function_11(theta_11, Z_trace, g, spl_neg, spl_pos, der_theta_F1, der_theta_F2, Lambda, theta, n):
    theta[n,1,1] = theta_11
    N_particles = Z_trace.shape[1]
    d = theta.shape[1]
    f = np.zeros((d,d))

    for i in range(0,N_particles):
        x_i = Z_trace[n,i,:d]
        #x_i = np.random.multivariate_normal(Z_trace[n,i,:d],0.01*np.identity(d))
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] > 0 :
            f += ( spl_pos(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_pos(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )
        if Z_trace[n,i,2] == g[n,i,0] and Z_trace[n,i,3] == g[n,i,1] and x_i[0] <= 0 :
            f += ( spl_neg(x_i[0], x_i[1], dx=1) * der_theta_F1(n, x_i[0], x_i[1], theta) +
                   spl_neg(x_i[0], x_i[1], dy=1) * der_theta_F2(n, x_i[0], x_i[1], theta)
                   )

    double = False
    if double == False:
        f3_tot = 2*Lambda[1,1]*theta[n,1,1] + f[1,1]/N_particles
    else:
        f3_tot = 2*Lambda[1,1]*theta[n,1,1] + 2 * f[1,1]/N_particles

    return f3_tot

def MFOC(N, d, T, dt, R, mu_0, center_left, center_right, y_left, y_right, xmin, xmax, grid_points, theta, F, mid_point, Lambda, num_iterations):
    # Creating the initial distribution based on the user's choice
    Z_all = initial_distribution(y_left, y_right, N, mu_0, d, R, mid_point, center_left, center_right)

    Nt = int(round(T/float(dt)))
    counts = np.zeros(num_iterations)
    thetas = np.zeros((num_iterations+1, Nt-1,d,d))
    thetas[0,:] = theta.reshape((Nt-1,d,d))

    for k in range(0,num_iterations):
        print(" --> ITERATION %s" %(k+1))

        # Resolution of the forward equation
        print("Moving the particles forward...")
        Z_trace, g = move_forward(dt, Nt, N, Z_all, F, theta, mid_point, y_left, y_right)

        #Resolution of the backward equation
        x = np.linspace(xmin, xmax, grid_points)
        Nx = grid_points
        dx = x[1] - x[0]
        x1 = np.linspace(xmin, xmax, Nx)
        x2 = np.linspace(xmin, xmax, Nx)
        y1 = np.linspace(xmin, xmax, Nx)
        y2 = np.linspace(xmin, xmax, Nx)
        print("Solving the backward equation...")
        psi = move_backward(x1, x2, y1, y2, xmin, xmax, dx, Nx, dt, Nt, I, F1, F2, theta, diff= "fu")

        #Extraction of the relevant information
        for i in range(0,Nx):
            if y1[i] == y_left[0]:
                n_neg_y1 = i
            if y1[i] == y_right[0]:
                n_pos_y1 = i
        for i in range(0,Nx):
            if y2[i] == y_left[1]:
                n_neg_y2 = i
            if y2[i] == y_right[1]:
                n_pos_y2 = i
        psi_neg = psi[:,:,:,n_neg_y1, n_neg_y2]
        psi_pos = psi[:,:,:,n_pos_y1, n_pos_y1]

        #Resolution of the parameter update
        print("Updating the parameter...")
        theta, count, loss_fct = parameter_update(Z_trace, g, psi_pos, psi_neg, der_theta_F1, der_theta_F2, theta, Lambda, x1 , x2, dx , dt, Nx, Nt, mid_point)
        counts[k] = count
        thetas[k+1,:] = theta.reshape((Nt-1,d,d))

    plt.figure()
    Z_trace, g = move_forward(dt, Nt, N, Z_all, F, thetas[-1,:].reshape((Nt-1,d,d)), mid_point, y_left, y_right)
    plt.scatter(Z_trace[0,:, 0], Z_trace[0,:,1], c='red', label='Moving points')
    plt.scatter(Z_trace[0,:, 2], Z_trace[0,:,3], c='green', label='Labels')
    plt.scatter(Z_trace[Nt-1,:, 0], Z_trace[Nt-1,:,1], c='blue', label='Moved points')
    plt.scatter(Z_trace[Nt-1,:, 2], Z_trace[Nt-1,:,3], c='green')
    plt.legend()
    plt.title("Plot of the points moving over time")
    plt.savefig("Particles_movement.png")
    plt.show()

    print("The loss fuction has value:")
    print(loss_fct)

    print("The number of sign switches for each iteration is: ")
    print(counts)

    return theta, thetas
