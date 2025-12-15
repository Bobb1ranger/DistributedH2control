import numpy as np
import control as ct
import os
from scipy.linalg import kron
from .Discrete_sim import simulate_discrete_ss
from H2control.feedback import lft
from H2control.DistributedH2controlTF import structured_output_feedback_h2
import H2control.DistributedH2controlTF as dh2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from .PID_compare import pid_to_discrete_ss



if __name__ == "__main__":
    # -----------------------------
# Graph / agent configuration
# -----------------------------
    adjPlant = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1]
    ])

    nAgents = 3

    Px = 3 * np.ones(nAgents, dtype=int)
    Py = np.ones(nAgents, dtype=int)
    Pu = np.ones(nAgents, dtype=int)

    # Directed graph and delay matrix

    delayMat = np.array([
        [0, -1, -1],
        [1, 0, -1],
        [2, 1, 0]
    ])

    # -----------------------------
    # Continuous-time agent model
    # -----------------------------
    tau = 0.1

    A0ct = np.array([
        [-1/tau,  0.0,  0.0],
        [ 1.0,    0.0,  0.0],
        [ 0.0,   -1.0,  0.0]
    ])

    B0ct = np.array([
        [ 1/tau,  0.0],
        [ 0.0,   -1.0],
        [ 0.0,    0.0]
    ])

    C0ct = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Continuous-time plant
    P0ct = ct.ss(A0ct, B0ct, C0ct, 0)

    # Discretization
    dt = 1/16
    P0dt = ct.c2d(P0ct, dt, method='zoh')

    A0dt = P0dt.A
    B0dt = P0dt.B

    # -----------------------------
    # Global system matrices
    # -----------------------------
    A = kron(np.eye(nAgents), A0dt)

    gamma1 = 1
    gamma2 = 5
    alpha  = 2

    # Output matrix C2
    C2 = kron(np.eye(nAgents), np.array([[0, 0, 1]]))

    for ii in range(1, nAgents):
        C2[ii, ii*3 - 1] = -1
    # print("C2:",C2)

    # Control input
    B2 = kron(np.eye(nAgents), B0dt[:, [0]])

    # Disturbance inputs
    Bw0  = kron(np.ones((nAgents, 1)), B0dt[:, [1]])
    Bw12 = np.hstack([B2 * gamma2, B2 * 0])

    B1 = np.hstack([Bw0, Bw12])

    # Performance output
    C1 = np.vstack([
        C2,
        np.zeros((nAgents, 3*nAgents))
    ])

    # Feedthrough matrices
    D11 = np.zeros((2*nAgents, 2*nAgents + 1))
    D12 = np.vstack([
        np.zeros((nAgents, nAgents)),
        np.eye(nAgents) * alpha
    ])

    D21 = np.hstack([
        np.zeros((nAgents, nAgents + 1)),
        np.eye(nAgents) * gamma1
    ])

    D22 = np.zeros((nAgents, nAgents))
    P_platoon = dh2.DiscreteGeneralizedPlant(A, B1, B2, C1, C2, D11, D12, D21, D22, dt)
    Kopt = structured_output_feedback_h2(P_platoon, delayMat, tol = 1e-4)


    ## Redefining system matrices, removing weights
    Bw0  = kron(np.ones((nAgents, 1)), B0dt[:, [1]])
    Bw12 = np.hstack([B2, B2 * 0])

    B1 = np.hstack([Bw0, Bw12])

    # Performance output
    C1 = np.vstack([
        C2,
        np.zeros((nAgents, 3*nAgents))
    ])

    # Feedthrough matrices
    D11 = np.zeros((2*nAgents, 2*nAgents + 1))
    D12 = np.vstack([
        np.zeros((nAgents, nAgents)),
        np.eye(nAgents)
    ])

    D21 = np.hstack([
        np.zeros((nAgents, nAgents + 1)),
        np.eye(nAgents)
    ])

    D22 = np.zeros((nAgents, nAgents))

    # Horizontal concatenation
    B = np.hstack([B1, B2])

    # Vertical concatenation
    C = np.vstack([C1, C2])

    # Block matrix for D
    top = np.hstack([D11, D12])
    bottom = np.hstack([D21, D22])
    D = np.vstack([top, bottom])
    P_platoon = ct.ss(A, B, C, D, dt)
    sys_cl = lft(P_platoon, Kopt)
    pPID = 0.16
    iPID = 0.005
    dPID = 0.1
    nPID = 2.5
    KPID_dt = pid_to_discrete_ss(pPID, iPID, dPID, nPID, dt, nAgents)
    # --- Example parameters ---
    sys_cl_PID = lft(P_platoon, KPID_dt)



    eigvals = np.linalg.eigvals(sys_cl_PID.A)
    print(eigvals)
    # exit()

    Tsim = 80.0          # total simulation time in seconds
    N = int(Tsim/dt)     # number of steps
    nx = sys_cl.A.shape[0]  # state dimension
    nz = sys_cl.C.shape[0]  # output dimension
    nw = B1.shape[1]     # process noise channels
    d = 10.0
    # Initial condition
    x0 = np.zeros(nx)

    # --- Generate process noise w ---
    np.random.seed(23)
    w = np.random.normal(0, 0.4*dt, size=(N, nw))  # small white noise
    impulse_index = int(1/dt)
    w[impulse_index, 0] = -1.0  # first channel impulse at t=1s
    w[impulse_index + 1, 0] = 1.0

    # Simulate
    x = np.zeros((N+1, nx))
    z = np.zeros((N, nz))
    x[0] = x0
    # w0: disturbance on car0's acceleration/speed (shape N,)
    delta_v = np.zeros(N+1)  # velocity change
    delta_z = np.zeros(N+1)  # position change

    for k in range(1, N+1):
        delta_v[k] = delta_v[k-1] + w[k-1,0] * dt      # integrate acceleration -> velocity
        delta_z[k] = delta_z[k-1] + delta_v[k] * dt   # integrate velocity -> position

    # Remove the first element to match z shape
    delta_z = delta_z[1:]  # shape (N,)
    for k in range(N):
        z[k] = sys_cl.C @ x[k] + sys_cl.D @ w[k]
        x[k+1] = sys_cl.A @ x[k] + sys_cl.B @ w[k]

    z_rel = z[:, :3]  # shape (N, 3)

    # Compute cumulative distance relative to car 0
    z_cumulative = np.zeros_like(z_rel)
    z_cumulative[:, 0] = z_rel[:, 0] + d + delta_z           # first car relative to reference
    z_cumulative[:, 1] = z_rel[:, 0] + z_rel[:, 1] + 2*d + delta_z  # second car
    z_cumulative[:, 2] = z_rel[:, 0] + z_rel[:, 1] + z_rel[:, 2] + 3*d + delta_z  # third car
    # # --- Plot results ---


    
    N = z_cumulative.shape[0]
    cars_z_top = np.zeros((N, 4))
    cars_z_top[:, 0] = delta_z          # reference car 0
    cars_z_top[:, 1:4] = z_cumulative[:, :3]

    # PID controller simulation
    nx = sys_cl_PID.A.shape[0]  # state dimension
    nz = sys_cl_PID.C.shape[0]  # output dimension
    x = np.zeros((N+1, nx))
    z = np.zeros((N, nz))
    x0 = np.zeros(nx)

    x[0] = x0
    # w0: disturbance on car0's acceleration/speed (shape N,)
    delta_v = np.zeros(N+1)  # velocity change
    delta_z = np.zeros(N+1)  # position change

    for k in range(1, N+1):
        delta_v[k] = delta_v[k-1] + w[k-1,0] * dt      # integrate acceleration -> velocity
        delta_z[k] = delta_z[k-1] + delta_v[k] * dt   # integrate velocity -> position

    # Remove the first element to match z shape
    delta_z = delta_z[1:]  # shape (N,)
    for k in range(N):
        z[k] = sys_cl_PID.C @ x[k] + sys_cl_PID.D @ w[k]
        x[k+1] = sys_cl_PID.A @ x[k] + sys_cl_PID.B @ w[k]

    z_rel = z[:, :3]  # shape (N, 3)

    # Compute cumulative distance relative to car 0
    z_cumulative = np.zeros_like(z_rel)
    z_cumulative[:, 0] = z_rel[:, 0] + d + delta_z           # first car relative to reference
    z_cumulative[:, 1] = z_rel[:, 0] + z_rel[:, 1] + 2*d + delta_z  # second car
    z_cumulative[:, 2] = z_rel[:, 0] + z_rel[:, 1] + z_rel[:, 2] + 3*d + delta_z  # third car
    cars_z_pid = np.zeros((N, 4))
    cars_z_pid[:, 0] = delta_z          # reference car 0
    cars_z_pid[:, 1:4] = z_cumulative[:, :3]


    script_dir = os.path.dirname(os.path.abspath(__file__))
    car_path = os.path.join(script_dir, 'car.png')
    tree_path = os.path.join(script_dir, 'tree.png')

    car_icon = mpimg.imread(car_path)
    tree_icon = mpimg.imread(tree_path)

    N = cars_z_top.shape[0]
    num_cars = 4
    num_trees = 1
    tree_speed = 10 * dt

    # Initialize figure with two vertical subplots
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 6))

    for ax in [ax_top, ax_bot]:
        ax.set_xlim(-5, 3 * d + 10)  # adjust depending on your scale
        ax.set_ylim(0, 5)
        ax.set_xlabel('Z (distance)')
        ax.set_ylabel('Y')
        
    ax_top.set_title(
        "Distributed H2 Controller",
        fontsize=16,
        fontweight='bold'
    )

    ax_bot.set_title(
        "PID Controller",
        fontsize=16,
        fontweight='bold'
    )
    ghost_imgs = []
    for i in range(num_cars):
        im = ax_top.imshow(car_icon, extent=[cars_z_top[0,i]-1, cars_z_top[0,i]+1, 0, 2], alpha=0.3, zorder=1)
        ghost_imgs.append(im)
        im = ax_bot.imshow(car_icon, extent=[cars_z_pid[0,i]-1, cars_z_pid[0,i]+1, 0, 2], alpha=0.3, zorder=1)
        ghost_imgs.append(im)

    # --- Initialize cars ---
    cars_imgs_top = [ax_top.imshow(car_icon, extent=[0,1,0,2], alpha=0, zorder=5) for _ in range(num_cars)]
    cars_imgs_bot = [ax_bot.imshow(car_icon, extent=[0,1,0,2], alpha=0, zorder=5) for _ in range(num_cars)]

    # --- Initialize trees ---
    tree_x = np.linspace(0, ax_top.get_xlim()[1], num_trees)
    tree_imgs_top = [ax_top.imshow(tree_icon, extent=[tree_x[i]-1, tree_x[i]+1, 0, 3], alpha=0, zorder=0) for i in range(num_trees)]
    tree_imgs_bot = [ax_bot.imshow(tree_icon, extent=[tree_x[i]-1, tree_x[i]+1, 0, 3], alpha=0, zorder=0) for i in range(num_trees)]

    # --- Animation function ---
    def animate(k):
        # Update cars
        for i in range(num_cars):
            cars_imgs_top[i].set_extent([cars_z_top[k,i]-1, cars_z_top[k,i]+1, 0, 2])
            if k > 0:
                cars_imgs_top[i].set_alpha(1)
            cars_imgs_bot[i].set_extent([cars_z_pid[k,i]-1, cars_z_pid[k,i]+1, 0, 2])
            if k > 0:
                cars_imgs_bot[i].set_alpha(1)
            
        # Update trees
        for i in range(num_trees):
            # Top subplot
            if tree_imgs_top[i].get_extent()[0] + tree_speed > ax.get_xlim()[1]:
                x0_top = (tree_imgs_top[i].get_extent()[0] + tree_speed) - ax.get_xlim()[1] + ax.get_xlim()[0]
            else:
                x0_top = tree_imgs_top[i].get_extent()[0] + tree_speed
            x1_top = x0_top + 2
            tree_imgs_top[i].set_extent([x0_top, x1_top, 0, 3])
            if k > 0:
                tree_imgs_top[i].set_alpha(0.2)
            
            # Bottom subplot
            if tree_imgs_bot[i].get_extent()[0] + tree_speed > ax.get_xlim()[1]:
                x0_bot = (tree_imgs_bot[i].get_extent()[0] + tree_speed) - ax.get_xlim()[1] + ax.get_xlim()[0]
            else:
                x0_bot = tree_imgs_bot[i].get_extent()[0] + tree_speed
            x1_bot = x0_bot + 2
            tree_imgs_bot[i].set_extent([x0_bot, x1_bot, 0, 3])
            if k > 0:
                tree_imgs_bot[i].set_alpha(0.2)
        
        return cars_imgs_top + cars_imgs_bot + tree_imgs_top + tree_imgs_bot
    frames_plot = range(0, N, 10)
    anim = FuncAnimation(fig, animate, frames = frames_plot, interval = dt*1000/4, blit=True)
    
    plt.tight_layout()
    plt.show()
    anim.save('platoon_animation.gif', writer='pillow', fps=20)