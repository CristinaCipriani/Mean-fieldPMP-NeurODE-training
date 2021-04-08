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
        X_all = np.concatenate([X_left,X_right], axis=0)
        Z_all = np.concatenate([Z_left,Z_right], axis=0)

    return Z_all

def move_forward(dt, Nt, N_particles, Z, F, theta, mid_point, y_left, y_right, xmin, xmax):
    T, d  = theta.shape

    Z_trace = np.zeros(shape=(Nt,2*N_particles,2*d))
    Z_trace[0,:,:] = Z

    g = np.zeros(shape=(Nt,2*N_particles,d))
    g[0,:] = y_right * np.sign(Z[:,0][:, np.newaxis])

    for n in range(0, Nt-1):
        Z_trace[n+1,:,:d] = Z_trace[n,:,:d] + dt * F(Z_trace[n,:,:d], theta[n, :]) #movement forward of x
        Z_trace[n+1,:,d:] = Z_trace[n,:,d:]                                        #movemente forward of y

        g[n+1,:] = y_right * np.sign(Z_trace[n+1,:,:d])        # vector that saves the actual sign of the particles

    return Z_trace, g

def move_backward(x, y, xmin, xmax, Nx, Ny, dt, Nt, F, theta, diff):
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    u = np.zeros((Nx+2, Ny+2)) # unknown u at new time level enlarged to contain the two ghost points
    u_n = np.zeros((Nx+2, Ny+2)) # u at the previous time level enlarged to contain the two ghost points
    u_all = np.zeros((Nt, Nx, Ny))

    x_b = np.linspace(xmin-dx, xmax + dx, Nx+2)  #here are the two ghost points, xmin-dx and xmax +dx
    y_b = np.linspace(xmin-dx, xmax + dx, Ny+2)

    # Load initial condition into u_n
    for i in range(0, Nx+2):
        for j in range(0, Ny+2):
             u_n[i,j] = np.abs(x_b[i]-y_b[j])**2
    u_all[Nt-1,:,:] = u_n[1:-1,1:-1]   #no need to save the ghost points

    for n in range(0, Nt-1):
        if diff == 'fu':
          # Boundary conditions
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
          # Internal points
          for j in range(0,Ny+2):
            u[1:Nx+1, j] = ( u_n[1:Nx+1,j]
                               - (-F(x_b[1:Nx+1], theta[Nt-2-n,:]))*dt/dx * (-u_n[1:Nx+1,j] + u_n[2:Nx+2,j])
                                 * 0.5*(1-np.sign(-F(x_b[0:Nx], theta[Nt-2-n,:])))
                               - (-F(x_b[0:Nx], theta[Nt-2-n,:]))*dt/dx * (-u_n[0:Nx,j] + u_n[1:Nx+1,j])
                                 * 0.5*(1+np.sign(-F(x_b[1:Nx+1], theta[Nt-2-n,:])))
                               )

        u_all[Nt-2-n,:,:] = u[1:-1,1:-1]
        # Switch variables before next step
        u_n, u = u, u_n

    return u_all


def parameter_update(Z_trace, g, psi_pos, psi_neg, F, theta, Lambda, x , dx , dt, Nx, Nt, mid_point, dynamics_plots):
    d = 1
    theta_new = np.zeros((Nt-1,d))
    f_value = np.zeros(Nt-1)
    f_0 = 0
    f_prime_0 = 0

    count = 0
    loss_fct = 0

    N_particles = Z_trace.shape[1]

    if dynamics_plots:
        fig, axs = plt.subplots(1, Nt-1, figsize=(35,5) )
    for n in range(0, Nt-1):
        spl_neg = interpolate.BSpline(x, psi_neg[n,:], k=2)
        spl_pos = interpolate.BSpline(x, psi_pos[n,:], k=2)

        # Checking the function shape before finding the root
        theta_domain = np.linspace(-30,30, 1001)
        f_domain = np.zeros(1001)

        if dynamics_plots:
            for it in range(len(theta_domain)):
                f_domain[it]= root_function_th(theta_domain[it], Z_trace, g, spl_neg, spl_pos, F, Lambda, Nt, d, n)
            axs[n].plot(theta_domain, f_domain)
            axs[n].plot(theta_domain, np.zeros(1001), 'r')

        sol = optimize.brenth(root_function_th, -200, 200, args=(Z_trace, g, spl_neg, spl_pos, F, Lambda, Nt, d, n))
        theta_new[n,:] = sol

        if dynamics_plots:
            axs[n].scatter(sol, root_function_th(sol, Z_trace, g, spl_neg, spl_pos, F, Lambda, Nt, d, n))

        #Updating f_value and count
        theta_ext = sol * np.ones((Nt-1,d))
        f_value[n], _, c = root_function(Z_trace, g, spl_neg, spl_pos, F, Lambda, theta_ext, n , False)
        count += c


    print("With a mean accuracy of %s, for each time step, theta has values:" %np.mean(f_value))
    theta_r = np.around(theta_new, 3)
    print (*theta_r, sep='  ')
    if dynamics_plots:
        fig.tight_layout()
        plt.show()
    print("While the total number of times in which there was a switch sign is %s" %count)

    # Calculating the loss function
    loss_fct = 0
    for i in range(0,N_particles):
        loss_fct += (1/N_particles) * np.abs(Z_trace[Nt-2,i,0]-Z_trace[Nt-2,i,1])**2
    loss_fct += Lambda * dt * np.linalg.norm(theta)**2

    return theta_new, f_value, count, loss_fct

def root_function(Z_trace, g, spl_neg, spl_pos, F, Lambda, theta, n, printy):
    N_particles = Z_trace.shape[1]
    count = 0
    f_0 = 0
    f_prime_0 = 0

    for i in range(0,N_particles):
        x_i = Z_trace[n,i,0]
        if Z_trace[n,i,1] == g[n,i] and x_i > 0 :
            f_0 += spl_pos(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * x_i
            f_prime_0 += spl_pos(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * np.tanh(theta[n,:]*x_i) * x_i**2

        if Z_trace[n,i,1] == g[n,i] and x_i <= 0 :
            prova = 2*(1-0.5*F(x_i, theta[n,:])**2) * x_i
            f_0 += spl_neg(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * x_i
            if printy == True:
                print("Currently processing n=%s : x_i is %s, grad of psi = %s and der of F = %s " %(n, x_i, spl_neg(x_i,nu=1), prova))
                print("And theta[n,:] is %s" %theta[n,:])
            f_prime_0 += spl_neg(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * np.tanh(theta[n,:]*x_i) * x_i**2

        if Z_trace[n,i,1] != g[n,i] :
            count += 1
            if Z_trace[n,i,1] >= 0:
                f_0 += spl_pos(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * x_i
                f_prime_0 += spl_pos(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * np.tanh(theta[n,:]*x_i) * x_i**2
            else:
                f_0 += spl_neg(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * x_i
                f_prime_0 += spl_neg(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * np.tanh(theta[n,:]*x_i) * x_i**2

    double = False
    if double == True:
        f_tot = 2*Lambda*theta[n,:] + f_0/N_particles
        f_prime_tot = 2*Lambda - (2*f_prime_0)/N_particles
    else:
        f_tot = 2*Lambda*theta[n,:] + (2*f_0)/N_particles
        f_prime_tot = 2*Lambda - (4*f_prime_0)/N_particles

    return f_tot, f_prime_tot, count

def root_function_th(theta_n, Z_trace, g, spl_neg, spl_pos, F, Lambda, Nt, d, n):
    theta = np.zeros((Nt-1,d))
    theta[n,:] = theta_n
    N_particles = Z_trace.shape[1]
    count = 0
    f_0 = 0
    f_prime_0 = 0

    for i in range(0,N_particles):
        x_i = Z_trace[n,i,0]
        if Z_trace[n,i,1] == g[n,i] and x_i >= 0 :
            f_0 += spl_pos(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * x_i

        if Z_trace[n,i,1] == g[n,i] and x_i < 0 :
            f_0 += spl_neg(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * x_i

        if Z_trace[n,i,1] != g[n,i] :
            count += 1
            if Z_trace[n,i,1] >= 0:
                f_0 += spl_pos(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * x_i
            else:
                f_0 += spl_neg(x_i, nu=1) * (1-np.tanh(theta[n,:]*x_i)**2) * x_i

    double = False
    if double == False:
        f_tot = 2*Lambda*theta[n,:] + f_0/N_particles
    else:
        f_tot = 2*Lambda*theta[n,:] + (2*f_0)/N_particles

    return f_tot

def MFOC(N, d, T, dt, R, mu_0, center_left, center_right, y_left, y_right, xmin, xmax, grid_points, theta, F, mid_point, Lambda, num_iterations):
    # Decide if I'm going to plot the dynamics at each time step or not
    if dt < 0.05:
        dynamics_plots = False
        print("Not printing the dynamics at each time step because dt is too small and the plots would be too messy!")
    else:
        dynamics_plots = True

    # Creating the initial distribution according to the user's choice
    Z_all = initial_distribution(y_left, y_right, N, mu_0, d, R, mid_point, center_left, center_right)

    Nt = int(round(T/float(dt)))
    counts = np.zeros(num_iterations)
    thetas = np.zeros((num_iterations+1, Nt-1))
    thetas[0,:] = theta.T

    for k in range(0,num_iterations):
        print(" --> ITERATION %s" %(k+1))

        # Resolution of the forward equation
        print("Moving the particles forward...")
        Z_trace, g = move_forward(dt, Nt, N, Z_all, F, theta, mid_point, y_left, y_right, xmin, xmax)

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
            x_new = np.linspace(xmin+dx*20, xmax-dx*20, 200)
            x_new = np.linspace(-2.5, 2.5,200)
            for n in range(Nt-1):
                spl_n = interpolate.BSpline(x[20:-20],psi_neg[Nt-1-n,:,],k=2)
                spl_p = interpolate.BSpline(x[20:-20],psi_pos[Nt-1-n,:],k=2)
                axs[0,Nt-2-n].plot(x_new, spl_n(x_new,nu=1),'g')
                axs[1,Nt-2-n].plot(x_new, spl_p(x_new,nu=1),'g')
            fig.tight_layout()
            plt.show()

        #Resolution of the parameter update
        print("Updating the parameter...")
        theta, f_value, count, loss_fct = parameter_update(Z_trace, g, psi_pos, psi_neg, F, theta, Lambda, x[20:-20] , dx , dt, Nx, Nt, mid_point, dynamics_plots)
        counts[k] = count
        thetas[k+1,:] = theta.T

    plt.figure()
    Z_trace, g= move_forward(dt, Nt, N, Z_all, F, thetas[-1,:].reshape(thetas.shape[1],1), mid_point, y_left, y_right, xmin, xmax)
    plt.scatter(Z_trace[0,:, 0], np.zeros(2*N),c='red',label='Moving points')
    plt.scatter(Z_trace[0,:, 1], np.zeros(2*N),c='green',label='Labels')
    for n in range(1,Nt-1):
        plt.scatter(Z_trace[n,:, 0], n*dt*np.ones(2*N),c='red')
        plt.scatter(Z_trace[n,:, 1], n*dt*np.ones(2*N),c='green')
    plt.legend()
    plt.title("Plot of the points moving over time")
    plt.savefig("Particles_movement.png")
    plt.show()

    print("The loss fuction has value:")
    print(loss_fct)

    print("The number of sign switches for each iteration is: ")
    print(counts)

    return theta, thetas
