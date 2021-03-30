import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
from IPython import display
from scipy import stats
from scipy import interpolate
from scipy import optimize
from sklearn.neighbors import KernelDensity

#Functions
def trunc_gaussian(shape, R):
    X = np.random.normal(size = shape)
    for i in range(shape[0]):
        while np.linalg.norm(X[i, :]) > R:
            X[i, :] = np.random.normal(size = shape[1])
    return X

def initial_distribution(y_left, y_right, N, mu_0, d, R, mid_point, center_left, center_right):
    # Creation of the initial distribution
    Y_left = y_left*np.ones((N,1))
    Y_right = y_right*np.ones((N,1))
    
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
        #X_all = np.concatenate([X_left,X_right], axis=0)
        Z_all = np.concatenate([Z_left,Z_right], axis=0)
        
    return Z_all

def move_forward(dt, Nt, N_particles, Z, F, theta, mid_point, y_left, y_right):
    T, d, _  = theta.shape
    
    Z_trace = np.zeros(shape=(Nt,2*N_particles,2*d))
    Z_trace[0,:,:] = Z
    
    g = np.zeros(shape=(Nt,2*N_particles,d))
    for i in range(2*N_particles):
        if Z_trace[0,i,:d] > mid_point:
            g[0,i] = y_right
        else:
            g[0,i] = y_left
    
    for n in range(0, Nt-1):
        Z_trace[n+1,:,:d] = Z_trace[n,:,:d] + dt * F(Z_trace[n,:,:d], theta[n, :,:])      #movement forward of x
        Z_trace[n+1,:,d:] = Z_trace[n,:,d:]                                               #movement forward of y
        
        for i in range(2*N_particles):
            if Z_trace[n+1,i,:d] > mid_point :                         # vector that saves the actual sign of the particles  
                g[n+1,i] = y_right
            else:
                g[n+1,i] = y_left
                
    return Z_trace, g
def move_backward(x, y, xmin, xmax, Nx, Ny, dt, Nt, F, theta, diff):
    dx = x[1] - x[0]
    dy = y[1] - y[0] 
    
    u = np.zeros((Nx+2, Ny+2)) # unknown u at new time level enlarged to contain the two ghost points
    u_n = np.zeros((Nx+2, Ny+2)) # u at the previous time level enlarged to contain the two ghost points
    u_all = np.zeros((Nt, Nx, Ny))
  
    x_b = np.linspace(xmin - dx, xmax + dx, Nx+2)  #here are the two ghost points, xmin-dx and xmax +dx 
    y_b = np.linspace(xmin - dx, xmax + dx, Ny+2)
    
    # Load initial condition into u_n
    for i in range(0, Nx+2):
        for j in range(0, Ny+2):
             u_n[i,j] = np.abs(x_b[i]-y_b[j])**2 
    u_all[Nt-1,:,:] = u_n[1:-1,1:-1]   #no need to save the ghost points
    
    for n in range(0, Nt-1):
        if diff == 'fu': #this is the method (1) from the notes, check it out
            u[0,:] = (u_n[0,:] 
                      - (-F(x_b[0], theta[Nt-2-n,:]))*dt/dx * (3/2*u_n[0,:] - 2*u_n[1,:] + 1/2*u_n[2,:]) 
                      * 0.5*(1+np.sign(-F(x_b[0], theta[Nt-2-n,:])))
                      - (-F(x_b[0],theta[Nt-2-n,:]))*dt/dx * (u_n[1,:] - u_n[0,:] )
                      * 0.5*(1-np.sign(-F(x_b[0], theta[Nt-2-n,:])))
                     )
            
            u[Nx+1,:] = (u_n[Nx+1,:] 
                         - (-F(x_b[Nx],theta[Nt-2-n,:]))*dt/dx * (-u_n[Nx,:]+u_n[Nx+1,:])
                             * 0.5*(1+np.sign(-F(x_b[Nx+1], theta[Nt-2-n,:])))
                         - (-F(x_b[Nx+1], theta[Nt-2-n,:]))*dt/dx * (+2*u_n[Nx,:] - 3/2*u_n[Nx+1,:] -1/2*u_n[Nx-1,:]) 
                             * 0.5*(1-np.sign(-F(x_b[Nx], theta[Nt-2-n,:])))
                        )
            
            for j in range(0,Ny+2):
                u[1:Nx+1, j] = ( u_n[1:Nx+1,j] 
                               - (-F(x_b[1:Nx+1], theta[Nt-2-n,:]))*dt/dx * (-u_n[1:Nx+1,j] + u_n[2:Nx+2,j]) 
                                 * 0.5*(1-np.sign(-F(x_b[0:Nx], theta[Nt-2-n,:])))
                               - (-F(x_b[0:Nx], theta[Nt-2-n,:]))*dt/dx * (-u_n[0:Nx,j] + u_n[1:Nx+1,j]) 
                                 * 0.5*(1+np.sign(-F(x_b[1:Nx+1], theta[Nt-2-n,:])))
                               )
        if diff == 'fu2': #this is the method (2) from the notes, which is oscillating a lot
            u[0,:] = (u_n[0,:] 
                      - F(x_b[0], theta[Nt-2-n,:])*dt/dx * (-3/2*u_n[0,:] + 2*u_n[1,:] - 1/2*u_n[2,:]) 
                      * 0.5*(1+np.sign(-F(x_b[0], theta[Nt-2-n,:])))
                      - F(x_b[0],theta[Nt-2-n,:])*dt/dx * (u_n[0,:] - u_n[1,:] )
                      * 0.5*(1-np.sign(-F(x_b[0], theta[Nt-2-n,:])))
                     )
            
            u[Nx+1,:] = (u_n[Nx+1,:] 
                         - F(x_b[Nx],theta[Nt-2-n,:])*dt/dx * (u_n[Nx,:]-u_n[Nx+1,:])
                             * 0.5*(1+np.sign(-F(x_b[Nx+1], theta[Nt-2-n,:])))
                         - F(x_b[Nx+1], theta[Nt-2-n,:])*dt/dx * (-2*u_n[Nx,:] + 3/2*u_n[Nx+1,:] +1/2*u_n[Nx-1,:]) 
                             * 0.5*(1-np.sign(-F(x_b[Nx], theta[Nt-2-n,:])))
                        )
            
            for j in range(0,Ny+2):
                u[1:Nx+1, j] = ( u_n[1:Nx+1,j] 
                               - F(x_b[1:Nx+1], theta[Nt-2-n,:])*dt/dx * (u_n[1:Nx+1,j] - u_n[2:Nx+2,j]) 
                                 * 0.5*(1-np.sign(-F(x_b[0:Nx], theta[Nt-2-n,:])))
                               - F(x_b[0:Nx], theta[Nt-2-n,:])*dt/dx * (u_n[0:Nx,j] - u_n[1:Nx+1,j]) 
                                 * 0.5*(1+np.sign(-F(x_b[1:Nx+1], theta[Nt-2-n,:])))
                               )
        if diff == 'fu_boh': #this is the method (2) from the notes, which is oscillating a lot
            u[0,:] = (u_n[0,:] 
                      - F(x_b[0], theta[Nt-2-n,:])*dt/dx * (+3/2*u_n[0,:] - 2*u_n[1,:] + 1/2*u_n[2,:]) 
                      * 0.5*(1+np.sign(-F(x_b[0], theta[Nt-2-n,:])))
                      - F(x_b[0],theta[Nt-2-n,:])*dt/dx * (u_n[0,:] - u_n[1,:] )
                      * 0.5*(1-np.sign(-F(x_b[0], theta[Nt-2-n,:])))
                     )
            
            u[Nx+1,:] = (u_n[Nx+1,:] 
                         - F(x_b[Nx],theta[Nt-2-n,:])*dt/dx * (u_n[Nx,:]-u_n[Nx+1,:])
                             * 0.5*(1+np.sign(-F(x_b[Nx+1], theta[Nt-2-n,:])))
                         - F(x_b[Nx+1], theta[Nt-2-n,:])*dt/dx * (-2*u_n[Nx,:] + 3/2*u_n[Nx+1,:] +1/2*u_n[Nx-1,:]) 
                             * 0.5*(1-np.sign(-F(x_b[Nx], theta[Nt-2-n,:])))
                        )
            
            for j in range(0,Ny+2):
                u[1:Nx+1, j] = ( u_n[1:Nx+1,j] 
                               - F(x_b[1:Nx+1], theta[Nt-2-n,:])*dt/dx * (u_n[1:Nx+1,j] - u_n[2:Nx+2,j]) 
                                 * 0.5*(1-np.sign(-F(x_b[0:Nx], theta[Nt-2-n,:])))
                               - F(x_b[0:Nx], theta[Nt-2-n,:])*dt/dx * (u_n[0:Nx,j] - u_n[1:Nx+1,j]) 
                                 * 0.5*(1+np.sign(-F(x_b[1:Nx+1], theta[Nt-2-n,:])))
                               )
          
        if diff == 'fu_without_Ftilda':
            u[0,:] = (u_n[0,:] 
                      + F(x_b[0], theta[Nt-2-n,:])*dt/dx * (-3/2*u_n[0,:] + 2*u_n[1,:] - 1/2*u_n[2,:]) * 0.5*(1+np.sign(F(x_b[0], theta[Nt-2-n,:])))
                      + F(x_b[0],theta[Nt-2-n,:])*dt/dx * (-u_n[1,:] + u_n[0,:] )
                      * 0.5*(1-np.sign(F(x_b[0], theta[Nt-2-n,:])))
                     )
            
            u[Nx+1,:] = (u_n[Nx+1,:] 
                         + F(x_b[Nx],theta[Nt-2-n,:])*dt/dx * (u_n[Nx,:]-u_n[Nx+1,:])
                             * 0.5*(1+np.sign(F(x_b[Nx+1], theta[Nt-2-n,:])))
                         + F(x_b[Nx+1], theta[Nt-2-n,:])*dt/dx * (-2*u_n[Nx,:] + 3/2*u_n[Nx+1,:] +1/2*u_n[Nx-1,:]) 
                             * 0.5*(1-np.sign(F(x_b[Nx], theta[Nt-2-n,:])))
                        )
            
            for j in range(0,Ny+2):
                u[1:Nx+1, j] = ( u_n[1:Nx+1,j] 
                               + F(x_b[1:Nx+1], theta[Nt-2-n,:])*dt/dx * (u_n[1:Nx+1,j] - u_n[2:Nx+2,j]) 
                                 * 0.5*(1-np.sign(F(x_b[0:Nx], theta[Nt-2-n,:])))
                               + F(x_b[0:Nx], theta[Nt-2-n,:])*dt/dx * (u_n[0:Nx,j] - u_n[1:Nx+1,j]) 
                                 * 0.5*(1+np.sign(F(x_b[1:Nx+1], theta[Nt-2-n,:])))
                               )
        if diff == 'fu_old':
            # Boundary conditions
            '''
            u[0,:] = (u_n[0,:] 
                      + F(x_b[0], theta[Nt-2-n,:])*dt/dx * (u_n[1,:] - u_n[0,:]) * 0.5*(1-np.sign(F(x_b[0], theta[Nt-2-n,:])))
                      + F(x_b[0],theta[Nt-2-n,:])*dt/dx * (2*u_n[1,:] -3/2*u_n[0,:] -1/2*u_n[2,:])
                      * 0.5*(1+np.sign(F(x_b[0], theta[Nt-2-n,:])))
                     )
            u[Nx+1,:] = (u_n[Nx+1,:] 
                         + F(x_b[Nx+1],theta[Nt-2-n,:])*dt/dx * (3/2*u_n[Nx+1,:]- 2*u_n[Nx,:] +1/2*u_n[Nx-1,:])
                                  * 0.5*(1-np.sign(F(x_b[Nx+1], theta[Nt-2-n,:])))
                         + F(x_b[Nx+1], theta[Nt-2-n,:])*dt/dx * (u_n[Nx+1,:] - u_n[Nx,:]) 
                                  * 0.5*(1+np.sign(F(x_b[Nx+1], theta[Nt-2-n,:])))
                        )
            u[0,:] = u_n[0,:] + F(x_b[0],theta[Nt-2-n,:])*dt/dx * (2*u_n[1,:] -3/2*u_n[0,:] -1/2*u_n[2,:])
            u[Nx+1,:] = u_n[Nx+1,:] + F(x_b[Nx+1],theta[Nt-2-n,:])*dt/dx * (-3/2*u_n[Nx+1,:]+ 2*u_n[Nx,:] -1/2*u_n[Nx-1,:])
            '''
            u[0,:] = (u_n[0,:] 
                      - F(x_b[0], theta[Nt-2-n,:])*dt/dx * (u_n[0,:] - u_n[1,:]) * 0.5*(1+np.sign(F(x_b[0], theta[Nt-2-n,:])))
                      + F(x_b[0],theta[Nt-2-n,:])*dt/dx * (-2*u_n[1,:] +3/2*u_n[0,:] +1/2*u_n[2,:])
                      * 0.5*(1-np.sign(F(x_b[0], theta[Nt-2-n,:])))
                     )
            u[Nx+1,:] = (u_n[Nx+1,:] 
                         - F(x_b[Nx+1],theta[Nt-2-n,:])*dt/dx * (3/2*u_n[Nx+1,:]- 2*u_n[Nx,:] +1/2*u_n[Nx-1,:])
                                  * 0.5*(1+np.sign(F(x_b[Nx+1], theta[Nt-2-n,:])))
                         + F(x_b[Nx+1], theta[Nt-2-n,:])*dt/dx * (u_n[Nx,:] - u_n[Nx+1,:]) 
                                  * 0.5*(1-np.sign(F(x_b[Nx+1], theta[Nt-2-n,:])))
                        )
            
            for j in range(0,Ny+2):
                '''
                u[1:Nx+1, j] = ( u_n[1:Nx+1,j] 
                               + F(x_b[1:Nx+1], theta[Nt-2-n,:])*dt/dx * (u_n[1:Nx+1,j] - u_n[0:Nx,j]) 
                                  * 0.5*(1+np.sign(F(x_b[1:Nx+1], theta[Nt-2-n,:])))
                               + F(x_b[1:Nx+1], theta[Nt-2-n,:])*dt/dx * (u_n[2:Nx+2,j] - u_n[1:Nx+1,j]) 
                                  * 0.5*(1-np.sign(F(x_b[1:Nx+1], theta[Nt-2-n,:])))
                               )
                
                u[n+1:Nx+1-n, j] = ( u_n[n+1:Nx+1-n,j] 
                               + F(x_b[n+1:Nx+1-n], theta[Nt-2-n,:])*dt/dx * (u_n[n+1:Nx+1-n,j] - u_n[n+0:Nx-n,j]) 
                                  * 0.5*(1+np.sign(F(x_b[n+1:Nx+1-n], theta[Nt-2-n,:])))
                               + F(x_b[n+1:Nx+1-n], theta[Nt-2-n,:])*dt/dx * (u_n[n+2:Nx+2-n,j] - u_n[n+1:Nx+1-n,j]) 
                                  * 0.5*(1-np.sign(F(x_b[n+1:Nx+1-n], theta[Nt-2-n,:])))
                               )
                '''
                u[1:Nx+1, j] = ( u_n[1:Nx+1,j] 
                               - F(x_b[1:Nx+1], theta[Nt-2-n,:])*dt/dx * (u_n[1:Nx+1,j] - u_n[2:Nx+2,j]) 
                                  * 0.5*(1+np.sign(F(x_b[1:Nx+1], theta[Nt-2-n,:])))
                               + F(x_b[1:Nx+1], theta[Nt-2-n,:])*dt/dx * (u_n[0:Nx,j] - u_n[1:Nx+1,j]) 
                                  * 0.5*(1-np.sign(F(x_b[1:Nx+1], theta[Nt-2-n,:])))
                               )
                
        if diff == 'lw':
            # Boundary conditions
            u[0,:] = u_n[0,:] - F(x_b[0],theta[n,:])*(-dt)/dx * (2*u_n[1,:] -3/2*u_n[0,:] -1/2*u_n[2,:])
            u[Nx+1,:] = u_n[Nx+1,:] -F(x_b[Nx+1],theta[n,:])*(-dt)/dx * (3/2*u_n[Nx+1,:]- 2*u_n[Nx,:] +1/2*u_n[Nx-1,:])
            
            for j in range(0,Ny+2):
                u[1:Nx+1,j] = (u_n[1:Nx+1,j] 
                              - (F(x_b[1:Nx+1], theta[n,:])*(-dt))/(2*dx) * (u_n[2:Nx+2,j]- u_n[0:Nx,j])
                              + (F(x_b[1:Nx+1], theta[n,:])*(-dt))**2/(2*dx**2) * (u_n[2:Nx+2,j] -2*u_n[1:Nx+1,j] + u_n[0:Nx,j])
                              )
                   
            
        u_all[Nt-2-n,:,:] = u[1:-1,1:-1]
        # Switch variables before next step
        u_n, u = u, u_n
        
    return u_all
    

def parameter_update(Z_trace, g, psi_pos, psi_neg, F, theta, Lambda, x , dx , dt, Nx, Nt, mid_point, dynamics_plots): 
    d = 1
    theta_new = np.zeros((Nt-1,d,2)) 
    f_value = np.zeros((Nt-1,2))
    
    count = 0
    loss_fct = 0
    
    N_particles = Z_trace.shape[1]

    if dynamics_plots:
        fig1, axs1 = plt.subplots(1, Nt-1, figsize=(35,5) )
        fig1.suptitle("Plot of root function of omega")
        fig2, axs2 = plt.subplots(1, Nt-1, figsize=(35,5) )
        fig2.suptitle("Plot of root function of sigma")
    for n in range(0, Nt-1):
        #spl_neg = interpolate.UnivariateSpline(x, psi_neg[n,:], k=2)   
        #spl_pos = interpolate.UnivariateSpline(x, psi_pos[n,:], k=2)
        spl_neg = interpolate.BSpline(x[20:-20], psi_neg[n,:,],k=2)
        spl_pos = interpolate.BSpline(x[20:-20], psi_pos[n,:], k=2)

        omega_domain = np.linspace(-15,15, 501)
        f0_domain = np.zeros(501)
        sigma_domain = np.linspace(-15,15, 501)
        f1_domain = np.zeros(501)

        sigma_n = theta[n,:,1]
        sol0 = optimize.brentq(root_function_omega, -200, 200, args=(sigma_n, Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n))
        f_value[n,0] = root_function_omega(sol0, sigma_n, Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n)
        theta_new[n,:,0] = sol0
        if dynamics_plots:
            axs1[n].scatter(sol0, root_function_omega(sol0, sigma_n, Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n))
            axs1[n].scatter(sol0, f_value[n,0])
            for it in range(len(omega_domain)):
                f0_domain[it] = root_function_omega(omega_domain[it], sigma_n ,Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n)
            axs1[n].plot(omega_domain, f0_domain)
            axs1[n].plot(omega_domain, np.zeros(501), 'r')
            
        omega_n = theta[n,:,0]
        sol1 = optimize.brentq(root_function_sigma, -20, 20, args=(omega_n, Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n))
        f_value[n,1] = root_function_sigma(sol1, omega_n, Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n)
        theta_new[n,:,1] = sol1
        if dynamics_plots:
            axs2[n].scatter(sol1, root_function_sigma(sol1, omega_n, Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n))
            axs2[n].scatter(sol1, f_value[n,1])
            for it in range(len(sigma_domain)):
                f1_domain[it] = root_function_sigma(sigma_domain[it], omega_n ,Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n)
            axs2[n].plot(sigma_domain, f1_domain)
            axs2[n].plot(sigma_domain, np.zeros(501), 'r')
        
        #Updating count
        count += root_function(Z_trace, g, spl_neg, spl_pos, F, Lambda, theta_new, mid_point, n, False)[4]

        '''
        # Checking the function shape before finding the root
        omega_domain = np.linspace(-5,5, 501)
        f0_domain = np.zeros(501)
        f0_prime_domain = np.zeros(501)
        
        sigma_domain = np.linspace(-5,5, 501)
        f1_domain = np.zeros(501)
        f1_prime_domain = np.zeros(501)
        
        theta_ext = theta
        for tt in range(len(omega_domain)):
            theta_ext[n,0,0] = omega_domain[tt] 
            f0_domain[tt],_, f0_prime_domain[tt],_,_ = root_function(Z_trace, g, spl_neg, spl_pos, F, Lambda, theta_ext, mid_point, n , False) #check this later
        
        theta_ext = theta
        for tt in range(len(sigma_domain)):
            theta_ext[n,0,1] = sigma_domain[tt]
            _,f1_domain[tt],_,f1_prime_domain[tt],_ = root_function(Z_trace, g, spl_neg, spl_pos, F, Lambda, theta_ext, mid_point, n , False) #check this later
     
        #f0_interp = interpolate.BSpline(omega_domain, f0_domain, k=4)    
        #f1_interp = interpolate.BSpline(sigma_domain, f1_domain, k=4)
        #f0_prime_interp = interpolate.BSpline(omega_domain, f0_prime_domain, k=4)
        #f1_prime_interp = interpolate.BSpline(sigma_domain, f1_prime_domain, k=4)
        if dynamics_plots:
            axs1[n].plot(omega_domain, f0_domain,'r') # or here .scatter()
            axs1[n].plot(omega_domain, np.zeros(501), 'b')
            #axs1[n].plot(omega_domain, f0_interp(omega_domain), 'g')
            axs2[n].plot(sigma_domain, f1_domain,'r') # or here .scatter()
            axs2[n].plot(sigma_domain, np.zeros(501), 'b')
            #axs2[n].plot(sigma_domain, f1_interp(sigma_domain), 'g')
                              
        #root_0 = optimize.root(f0_interp, theta[n,:,0])
        #root_1 = optimize.root(f1_interp, theta[n,:,1])
        #theta_new[n,:,0] = root_0.x
        #theta_new[n,:,1] = root_1.x
        #sol_0 = optimize.newton(f0_interp, theta[n,:,0])
        #sol_1 = optimize.newton(f1_interp, theta[n,:,1])
        sol_0 = optimize.brentq(f0_interp, a=-5, b=5)   #, theta[n,:,0], maxiter=500
        sol_1 = optimize.brentq(f1_interp, a=-5, b=5)   #, theta[n,:,1], maxiter=500
        theta_new[n,:,0] = sol_0
        theta_new[n,:,1] = sol_1 
        if dynamics_plots:
            axs1[n].scatter(theta_new[n,:,0], f0_interp(theta_new[n,:,0])) 
            axs2[n].scatter(theta_new[n,:,1], f1_interp(theta_new[n,:,1]))
         
        #Updating f_value and count
        f_value[n,0], f_value[n,1], _, _, c = root_function(Z_trace, g, spl_neg, spl_pos, F, Lambda, theta_new, mid_point, n, False)
        count += c
            
        # Finding the root
        itr0 = 0
        itr1 = 0
        f0 = 7
        f1 = 7
       
        while np.abs(f0) > 0.00000001 and itr0 < 1000:
            f0, _, f0_prime, _, _ = root_function(Z_trace, g, spl_neg, spl_pos, F, Lambda, theta, mid_point, n, False)
            theta_new[n,:,0] = theta[n,:,0] - f0*f0_prime/np.abs(f0_prime)**2
            theta[n,:,0] = theta_new[n,:,0]
            itr0 += 1
            
        while np.abs(f1) > 0.00000001 and itr1 < 1000:
            _, f1, _, f1_prime, _ = root_function(Z_trace, g, spl_neg, spl_pos, F, Lambda, theta, mid_point, n, False)
            theta_new[n,:,1] = theta[n,:,1] - f1*f1_prime/np.abs(f1_prime)**2
            theta[n,:,1] = theta_new[n,:,1]
            itr1 += 1
        
        if dynamics_plots:
            axs1[n].scatter(theta[n,:,0], f0_interp(theta[n,:,0]))
            axs1[n].set_title("Solution omega"+ str(np.around(theta[n,:,0],2)) +" after " + str(itr0)+ " iterations" )
            axs2[n].scatter(theta[n,:,1], f1_interp(theta[n,:,1]))
            axs2[n].set_title("Solution sigma"+ str(np.around(theta[n,:,1],2)) +" after " + str(itr1)+ " iterations" )
        f_value[n,0] = f0
        f_value[n,1] = f1
        '''
    
    print("With a mean accuracy of %s, for each time step, omega has values:" %np.mean(f_value[:,0]))
    omega_r = np.around(theta_new[:,:,0], 3)
    print (*omega_r, sep='  ')
    if dynamics_plots:     
        fig1.tight_layout()
    print("With a mean accuracy of %s, for each time step, sigma has values:" %np.mean(f_value[:,1]))
    sigma_r = np.around(theta_new[:,:,1], 3)
    print (*sigma_r, sep='  ')
    if dynamics_plots:     
        fig2.tight_layout()
        plt.show() 
    print("While the total number of times in which there was a switch sign is %s" %count)
    
    # Calculating the loss function
    loss_fct = 0
    for i in range(0,N_particles):
        loss_fct += (1/N_particles) * np.abs(Z_trace[Nt-2,i,0]-Z_trace[Nt-2,i,1])**2  
    loss_fct += Lambda[0] * dt * np.linalg.norm(theta_new[:,:,0])**2 + Lambda[1] * dt * np.linalg.norm(theta_new[:,:,1])**2
     
    return theta_new, f_value, count, loss_fct

def root_function_omega(omega_n, sigma_n, Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n):
    theta = np.zeros((Nt-1,d,2))
    theta[n,:,0] = omega_n
    theta[n,:,1] = sigma_n
    N_particles = Z_trace.shape[1]
    count = 0
    f_0 = 0
    f_prime_0 = 0

    for i in range(0,N_particles):
        x_i = Z_trace[n,i,0]
        #x_i = np.random.normal(Z_trace[n,i,0],0.1)     #this is if I sample the particles from a Gaussian (in the integral)
        if Z_trace[n,i,1] == g[n,i] and x_i > mid_point :
            f_0 += spl_pos(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2) * x_i
            f_prime_0 += spl_pos(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) * x_i**2
            
        if Z_trace[n,i,1] == g[n,i] and x_i <= mid_point :
            f_0 += spl_neg(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2) * x_i
            f_prime_0 += spl_neg(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) * x_i**2
            
        if Z_trace[n,i,1] != g[n,i] :
            count += 1
            if Z_trace[n,i,1] > mid_point:
                f_0 += spl_pos(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2) * x_i
                f_prime_0 += spl_pos(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) * x_i**2
            else:
                f_0 += spl_neg(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2) * x_i
                f_prime_0 += spl_neg(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) * x_i**2
    
    double = False 
    
    if double == False:
        f0_tot = 2*Lambda[0]*theta[n,:,0] + f_0/N_particles
        f0_prime_tot = 2*Lambda[0] - (2*f_prime_0)/N_particles
    else:
        f0_tot = 2*Lambda[0]*theta[n,:,0] + (2*f_0)/N_particles
        f0_prime_tot = 2*Lambda[0] - (4*f_prime_0)/N_particles
          
    return f0_tot

def root_function_sigma(sigma_n, omega_n, Z_trace, g, spl_neg, spl_pos, F, Nt, d, Lambda, mid_point, n):
    theta = np.zeros((Nt-1,d,2))
    theta[n,:,0] = omega_n
    theta[n,:,1] = sigma_n
    N_particles = Z_trace.shape[1]
    count = 0
    f_1 = 0
    f_prime_1 = 0
    
    for i in range(0,N_particles):
        x_i = Z_trace[n,i,0]
        #x_i = np.random.normal(Z_trace[n,i,0],0.1)     #this is if I sample the particles from a Gaussian (in the integral)
        if Z_trace[n,i,1] == g[n,i] and x_i > mid_point :
            f_1 += spl_pos(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2)
            f_prime_1 += spl_pos(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2)
            
        if Z_trace[n,i,1] == g[n,i] and x_i <= mid_point :
            f_1 += spl_neg(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2)
            f_prime_1 += spl_neg(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) 
            
        if Z_trace[n,i,1] != g[n,i] :
            count += 1
            if Z_trace[n,i,1] > mid_point:
                f_1 += spl_pos(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2)
                f_prime_1 += spl_pos(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) 
            else:
                f_1 += spl_neg(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2)
                f_prime_1 += spl_neg(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2)
    
    double = False 
    
    if double == False:
        f1_tot = 2*Lambda[1]*theta[n,:,1] + f_1/N_particles
        f1_prime_tot = 2*Lambda[1] - (2*f_prime_1)/N_particles
    else:
        f1_tot = 2*Lambda[1]*theta[n,:,1] + (2*f_1)/N_particles
        f1_prime_tot = 2*Lambda[1] - (4*f_prime_1)/N_particles
          
    return f1_tot

def root_function(Z_trace, g, spl_neg, spl_pos, F, Lambda, theta, mid_point, n, printy):
    N_particles = Z_trace.shape[1]
    count = 0
    f_0 = 0
    f_1 = 0
    f_prime_0 = 0
    f_prime_1 = 0
    
    for i in range(0,N_particles):
        x_i = Z_trace[n,i,0]
        #x_i = np.random.normal(Z_trace[n,i,0],0.1)     #this is if I sample the particles from a Gaussian (in the integral)
        if Z_trace[n,i,1] == g[n,i] and x_i > mid_point :
            f_0 += spl_pos(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2) * x_i
            f_1 += spl_pos(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2)
            f_prime_0 += spl_pos(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) * x_i**2
            f_prime_1 += spl_pos(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2)
            
        if Z_trace[n,i,1] == g[n,i] and x_i <= mid_point :
            f_0 += spl_neg(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2) * x_i
            f_1 += spl_neg(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2)
            f_prime_0 += spl_neg(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) * x_i**2
            f_prime_1 += spl_neg(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) 
            #if printy == True:
            #    print("Currently processing n=%s : x_i is %s, grad of psi = %s and der of F = %s " %(n, x_i, spl_neg(x_i,nu=1), prova))
            #    print("And theta[n,:] is %s" %theta[n,:])
            
        if Z_trace[n,i,1] != g[n,i] :
            count += 1
            if Z_trace[n,i,1] > mid_point:
                f_0 += spl_pos(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2) * x_i
                f_1 += spl_pos(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2)
                f_prime_0 += spl_pos(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) * x_i**2
                f_prime_1 += spl_pos(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) 
            else:
                f_0 += spl_neg(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2) * x_i
                f_1 += spl_neg(x_i, nu=1) * (1-F(x_i, theta[n,:,:])**2)
                f_prime_0 += spl_neg(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2) * x_i**2
                f_prime_1 += spl_neg(x_i, nu=1) * F(x_i, theta[n,:,:]) * (1-F(x_i, theta[n, :,:])**2)
    
    double = False 
    
    if double == False:
        f0_tot = 2*Lambda[0]*theta[n,:,0] + f_0/N_particles
        f0_prime_tot = 2*Lambda[0] - (2*f_prime_0)/N_particles
        f1_tot = 2*Lambda[1]*theta[n,:,1] + f_1/N_particles
        f1_prime_tot = 2*Lambda[1] - (2*f_prime_1)/N_particles
    else:
        f0_tot = 2*Lambda[0]*theta[n,:,0] + (2*f_0)/N_particles
        f0_prime_tot = 2*Lambda[0] - (4*f_prime_0)/N_particles
        f1_tot = 2*Lambda[1]*theta[n,:,1] + (2*f_1)/N_particles
        f1_prime_tot = 2*Lambda[1] - (4*f_prime_1)/N_particles
          
    return f0_tot, f1_tot, f0_prime_tot, f1_prime_tot, count    

def MFOC(N, d, T, dt, R, mu_0, center_left, center_right, y_left, y_right, xmin, xmax, grid_points, theta, F, mid_point, Lambda, num_iterations):
    # Decide if I'm going to plot the dynamics at each time step or not
    if dt < 0.05:
        dynamics_plots = False
        print("Not printing the dynamics at each time step because dt is too small and the plots would be too messy!")
    else:
        dynamics_plots = True
    
    Z_all = initial_distribution(y_left, y_right, N, mu_0, d, R, mid_point, center_left, center_right)
        
    Nt = int(round(T/float(dt)))
    counts = np.zeros(num_iterations)
    thetas = np.zeros((num_iterations+1, Nt-1,2))
    thetas[0,:,:] = theta.reshape((Nt-1,2))
    
    for k in range(0,num_iterations):
        print(" --> ITERATION %s" %(k+1))
        
        # Resolution of the forward equation
        print("Moving the particles forward...")
        Z_trace, g = move_forward(dt, Nt, N, Z_all, F, theta, mid_point, y_left, y_right)
        
        # Plot of the forward dynamics
        if dynamics_plots:
            fig, axs = plt.subplots(1, Nt-1, figsize=(35,5) )
            for n in range(Nt-1):
                for lines in range(n+1):
                    axs[n].set_ylim([-0.1, Nt*dt])
                    axs[n].scatter(Z_trace[lines,:, 0], lines*dt*np.ones(2*N),c='red')
                    axs[n].scatter(Z_trace[lines,:, 1], lines*dt*np.ones(2*N),c='green')
            fig.tight_layout()
            plt.show() 
    
        #Resolution of the backward equation
        x = np.linspace(xmin, xmax, grid_points)
        Nx = grid_points
        Ny = Nx
        y = np.linspace(xmin, xmax, Ny)
        dx = x[1]-x[0]
        print("Solving the backward equation...")
        psi = move_backward(x, y, xmin, xmax, Nx, Ny, dt, Nt, F, theta, diff="fu")
    
        #Extraction of the relevant information
        for j in range(0,Ny):
            if y[j] == y_left:
                n_neg = j
            if y[j] == y_right:
                n_pos = j
        psi_neg = psi[:,20:-20,n_neg]
        psi_pos = psi[:,20:-20,n_pos]
        
        #Plot of the backward dynamics
        if dynamics_plots:
            fig, axs = plt.subplots(2, Nt-1, figsize=(35,5) )
            #x_new = np.linspace(xmin+dx*20, xmax-dx*20, 200)
            x_new = np.linspace(-3,3,200)
            for n in range(Nt-1):
                #axs[0,Nt-2-n].scatter(x[20:-20],psi_neg[Nt-1-n,:])
                #axs[1,Nt-2-n].scatter(x[20:-20],psi_pos[Nt-1-n,:])
                spl_n = interpolate.BSpline(x[20:-20],psi_neg[Nt-1-n,:,],k=2)
                spl_p = interpolate.BSpline(x[20:-20],psi_pos[Nt-1-n,:,],k=2)
                #axs[0,Nt-2-n].plot(x_new, spl_n(x_new),'r')
                #axs[1,Nt-2-n].plot(x_new, spl_p(x_new),'r')
                axs[0,Nt-2-n].plot(x_new, spl_n(x_new,nu=1),'g')
                axs[1,Nt-2-n].plot(x_new, spl_p(x_new,nu=1),'g')
            fig.tight_layout()
            plt.show()

        #Resolution of the parameter update
        print("Updating the parameter...")
        theta, f_value, count, loss_fct = parameter_update(Z_trace, g, psi_pos, psi_neg, F, theta, Lambda, x , dx , dt, Nx, Nt, mid_point, dynamics_plots)
        counts[k] = count  
        thetas[k+1,:,:] = theta.reshape((Nt-1,2))
                
    plt.figure()
    Z_trace, g = move_forward(dt, Nt, N, Z_all, F, thetas[-1,:,:].reshape((Nt-1,d,2)), mid_point, y_left, y_right)
    plt.scatter(Z_trace[0,:, 0], np.zeros(2*N),c='red',label='Moving points')
    plt.scatter(Z_trace[0,:, 1], np.zeros(2*N),c='green',label='Labels')
    for n in range(1,Nt-1):
        plt.scatter(Z_trace[n,:, 0], n*dt*np.ones(2*N),c='red')
        plt.scatter(Z_trace[n,:, 1], n*dt*np.ones(2*N),c='green')
    plt.legend()
    plt.title("Plot of the points moving over time")
    plt.show()    
    
    print("The loss fuction has value:")
    print(loss_fct)
    
    print("The number of sign switches for each iteration is: ")
    print(counts)
    
    return theta, thetas