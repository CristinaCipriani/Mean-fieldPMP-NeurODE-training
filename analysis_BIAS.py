import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
from IPython import display
from scipy import stats
from scipy import interpolate
from sklearn.neighbors import KernelDensity
from modules.training_BIAS import initial_distribution, move_forward

def J(theta, Z_all, N, F, mid_point, y_left, y_right, dt, Lambda):
    T = 1
    Nt = int(round(T/float(dt)))
    d = 1
    Z_trace, _  = move_forward(dt, Nt, N, Z_all, F, theta, mid_point, y_left, y_right)
    
    loss_fct = 0
    for i in range(0,N):
        loss_fct += (1/N) * np.abs(Z_trace[Nt-2,i,0]-Z_trace[Nt-2,i,1])**2  
    loss_fct += Lambda[0] * dt * np.linalg.norm(theta[:,:,0])**2 + Lambda[1] * dt * np.linalg.norm(theta[:,:,1])**2
    
    return loss_fct
    
    
def plot_loss_fct(N, d, T, dt, R, mu_0, center_left, center_right, y_left, y_right, mid_point, F, Lambda, my_sol):
    Nt = int(round(T/float(dt)))
    
    Z_all = initial_distribution(y_left, y_right, N, mu_0, d, R, mid_point, center_left, center_right)
        
    omega_domain = np.linspace(-5,5, 1001)
    sigma_domain = np.linspace(-5,5, 1001)
    J_domain = np.zeros((1001,1001)) 
    
    for i in range(omega_domain.shape[0]) :
        for j in range(sigma_domain.shape[0]):
            theta_extended = np.zeros((Nt-1,d,2))
            theta_extended[:,:,0] = omega_domain[i] *np.ones((Nt-1,d))
            theta_extended[:,:,1] = sigma_domain[j] *np.ones((Nt-1,d))
            J_domain[i,j] = J(theta_extended, Z_all, N, F, mid_point, y_left, y_right, dt, Lambda)
        
    X, Y = np.meshgrid(omega_domain, sigma_domain)    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, J_domain, cmap='viridis', linewidth=0)
    plt.draw()
    
    J_interp = interpolate.RectBivariateSpline(omega_domain, sigma_domain, J_domain)
    #plt.scatter(np.mean(my_sol), J_interp(np.mean(my_sol)))
    idx_min = np.unravel_index(J_domain.argmin(), J_domain.shape)
    ax.scatter(omega_domain[idx_min[0]],sigma_domain[idx_min[1]], J_interp(omega_domain[idx_min[0]],sigma_domain[idx_min[1]]))
    #plt.show()
    #print("The value we found as minimum is (in mean) %s, corresponding to a loss of %s" %(np.mean(my_sol),J_interp(np.mean(my_sol))))
    print("While the actual mizimizer is (%s, %s), corresponding to a loss of %s" %(omega_domain[idx_min[0]], sigma_domain[idx_min[1]], J_interp(omega_domain[idx_min[0]], sigma_domain[idx_min[1]])))

    print("Let's try to move the points with the optimal theta to see if it actually works")
    theta_optimal = np.zeros((Nt-1,d,2)) 
    theta_optimal[:,:,0] = omega_domain[idx_min[0]] * np.ones((Nt-1,d))
    theta_optimal[:,:,1] = sigma_domain[idx_min[1]] * np.ones((Nt-1,d))
    Z_trace, _ = move_forward(dt, Nt, N, Z_all, F, theta_optimal, mid_point, y_left, y_right)
    plt.figure()
    plt.scatter(Z_trace[0,:, 0], np.zeros(2*N),c='red',label='Moving points')
    plt.scatter(Z_trace[0,:, 1], np.zeros(2*N),c='green',label='Labels')
    for n in range(1,Nt-1):
        plt.scatter(Z_trace[n,:, 0], n*dt*np.ones(2*N),c='red')
        plt.scatter(Z_trace[n,:, 1], n*dt*np.ones(2*N),c='green')
    plt.legend()
    plt.title("Plot of the points moving over time")
    plt.show()
    
    
def loss_comparison_Lambda(ax1, ax2, ax3, N, d, T, dt, R, mu_0, center_left, center_right, y_left, y_right, mid_point, F, Lambda, theta_limits, my_sol, points_movement):
    Nt = int(round(T/float(dt)))
    
    Z_all = initial_distribution(y_left, y_right, N, mu_0, d, R, mid_point, center_left, center_right)
        
    if theta_limits == None:
        theta_domain = np.linspace(-10,10, 1001)
    else:
        theta_domain = np.linspace(theta_limits[0], theta_limits[1],1001)
        
    J1_domain = np.zeros(1001) 
    J2_domain = np.zeros(1001)

    for j in range(0,theta_domain.shape[0]) :
        theta_extended = theta_domain[j] * np.ones((Nt-1,d)) 
        J1_domain[j] = J(theta_extended,Z_all, N, F, mid_point, y_left, y_right, dt, Lambda = 1)
        J2_domain[j] = J(theta_extended,Z_all, N, F, mid_point, y_left, y_right, dt, Lambda = 0.001)

    #ax1.plot(theta_domain,J1_domain,'r', label="Lambda = 1")
    #J1_interp = interpolate.BSpline(theta_domain, J1_domain, k=1)
    ax1.plot(theta_domain,J2_domain,'g', label="Lambda = 0.001")
    J2_interp = interpolate.BSpline(theta_domain, J2_domain, k=1)
    #ax1.scatter(theta_domain[np.argmin(J1_domain)], J1_interp(theta_domain[np.argmin(J1_domain)]))
    ax1.scatter(theta_domain[np.argmin(J2_domain)], J2_interp(theta_domain[np.argmin(J2_domain)]))
    ax1.legend()
    
    # plot of the movement of the points
    if points_movement == True:
        # Lamda = 1 
        theta_optimal = theta_domain[np.argmin(J1_domain)] * np.ones((Nt-1,d))
        Z_trace, _ = move_forward(dt, Nt, N, Z_all, F, theta_optimal, mid_point, y_left, y_right)
        ax2.scatter(Z_trace[0,:, 0], np.zeros(2*N),c='red',label='Moving points')
        ax2.scatter(Z_trace[0,:, 1], np.zeros(2*N),c='green',label='Labels')
        for n in range(1,Nt-1):
            ax2.scatter(Z_trace[n,:, 0], n*dt*np.ones(2*N),c='red')
            ax2.scatter(Z_trace[n,:, 1], n*dt*np.ones(2*N),c='green')
        ax2.legend()
        # Lamda = 0.1 
        theta_optimal = theta_domain[np.argmin(J2_domain)] * np.ones((Nt-1,d))
        Z_trace, _ = move_forward(dt, Nt, N, Z_all, F, theta_optimal, mid_point, y_left, y_right)
        ax3.scatter(Z_trace[0,:, 0], np.zeros(2*N),c='red',label='Moving points')
        ax3.scatter(Z_trace[0,:, 1], np.zeros(2*N),c='green',label='Labels')
        for n in range(1,Nt-1):
            ax3.scatter(Z_trace[n,:, 0], n*dt*np.ones(2*N),c='red')
            ax3.scatter(Z_trace[n,:, 1], n*dt*np.ones(2*N),c='green')
        ax3.legend()

    return 

def loss_comparison_dt(ax1, ax2, ax3, N, d, T, dt, R, mu_0, center_left, center_right, y_left, y_right, mid_point, F, Lambda, theta_limits, my_sol, points_movement):
    Z_all = initial_distribution(y_left, y_right, N, mu_0, d, R, mid_point, center_left, center_right)
        
    if theta_limits == None:
        theta_domain = np.linspace(-10,10, 1001)
    else:
        theta_domain = np.linspace(theta_limits[0], theta_limits[1],1001)
        
    J1_domain = np.zeros(1001) 
    J2_domain = np.zeros(1001)

    dt1 = 0.1
    dt2 = 0.01
    
    for j in range(0,theta_domain.shape[0]) :
        Nt1 = int(round(T/float(dt1)))
        Nt2 = int(round(T/float(dt2)))
        theta_extended1 = theta_domain[j] * np.ones((Nt1-1,d))
        theta_extended2 = theta_domain[j] * np.ones((Nt2-1,d))
        J1_domain[j] = J(theta_extended1,Z_all, N, F, mid_point, y_left, y_right, dt = dt1, Lambda = 1)
        J2_domain[j] = J(theta_extended2,Z_all, N, F, mid_point, y_left, y_right, dt = dt2, Lambda = 1)
    
    #plot of the loss function
    ax1.plot(theta_domain,J1_domain,'r', label="dt = 0.1")
    J1_interp = interpolate.BSpline(theta_domain, J1_domain, k=1)
    ax1.plot(theta_domain,J2_domain,'g', label="dt = 0.01")
    J2_interp = interpolate.BSpline(theta_domain, J2_domain, k=1)
    ax1.scatter(theta_domain[np.argmin(J1_domain)], J1_interp(theta_domain[np.argmin(J1_domain)]))
    ax1.scatter(theta_domain[np.argmin(J2_domain)], J2_interp(theta_domain[np.argmin(J2_domain)]))
    ax1.legend()
    
    # plot of the movement of the points
    if points_movement == True:
        # dt = 0.1 
        theta_optimal = theta_domain[np.argmin(J1_domain)] * np.ones((Nt1-1,d))
        Z_trace, _ = move_forward(dt1, Nt1, N, Z_all, F, theta_optimal, mid_point, y_left, y_right)
        ax2.scatter(Z_trace[0,:, 0], np.zeros(2*N),c='red',label='Moving points')
        ax2.scatter(Z_trace[0,:, 1], np.zeros(2*N),c='green',label='Labels')
        for n in range(1,Nt1-1):
            ax2.scatter(Z_trace[n,:, 0], n*dt1*np.ones(2*N),c='red')
            ax2.scatter(Z_trace[n,:, 1], n*dt1*np.ones(2*N),c='green')
        ax2.legend()
        # dt = 0.01 
        theta_optimal = theta_domain[np.argmin(J2_domain)] * np.ones((Nt2-1,d))
        Z_trace, _ = move_forward(dt2, Nt2, N, Z_all, F, theta_optimal, mid_point, y_left, y_right)
        ax3.scatter(Z_trace[0,:, 0], np.zeros(2*N),c='red',label='Moving points')
        ax3.scatter(Z_trace[0,:, 1], np.zeros(2*N),c='green',label='Labels')
        for n in range(1,Nt2-1):
            ax3.scatter(Z_trace[n,:, 0], n*dt2*np.ones(2*N),c='red')
            ax3.scatter(Z_trace[n,:, 1], n*dt2*np.ones(2*N),c='green')
        ax3.legend()
    
    return 
