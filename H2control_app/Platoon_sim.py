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
    print("C2:",C2)

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
    Kopt = structured_output_feedback_h2(P_platoon, delayMat, tol = 1e-6)


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
    P = ct.ss(A, B, C, D, dt)
    sys_cl = lft(P, Kopt)

    # --- Example parameters ---

    Tsim = 20.0          # total simulation time in seconds
    N = int(Tsim/dt)     # number of steps
    nx = sys_cl.A.shape[0]  # state dimension
    nz = sys_cl.C.shape[0]  # output dimension
    nw = B1.shape[1]     # process noise channels
    d = 10.0
    # Initial condition
    x0 = np.zeros(nx)

    # --- Generate process noise w ---
    np.random.seed(42)
    w = np.random.normal(0, 0.2, size=(N, nw))  # small white noise
    impulse_index = int(1/dt)
    w[impulse_index, 0] = -0.05  # first channel impulse at t=1s
    w[impulse_index + 1, 0] = 0.06

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
    # t = np.arange(N)*dt
    # plt.figure(figsize=(8,4))

    # for i in range(3):
    #     plt.plot(t, z_cumulative[:, i], label=f'z_cumulative[{i}]')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Output')
    # plt.title('Platoon system outputs with noise')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # --- Example: z_cumulative from your simulation ---
    # N time steps, 4 cars
    # For demo, let's assume z_cumulative has 3 cars (1st car relative to 0)
    # We'll add reference car 0 at z=0

    N = z_cumulative.shape[0]
    cars_z = np.zeros((N, 4))
    cars_z[:, 0] = delta_z          # reference car 0
    cars_z[:, 1:4] = z_cumulative[:, :3]

    # --- Load car icon ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    car_path = os.path.join(script_dir, 'car.png')
    car_icon = mpimg.imread(car_path) # Make sure the file exists

    # --- Setup figure ---
    fig, ax = plt.subplots(figsize=(8, 16))
    ax.set_xlim(-5, 3 * d + 10)  # x fixed
    ax.set_ylim(0, 5)
    ax.set_xlabel('Z (distance substract nominal translation)')
    ax.set_title('Platoon Cars Animated')
    ghost_imgs = []

    for i in range(4):
        im = ax.imshow(car_icon, extent=[cars_z[0,i]-1, cars_z[0,i]+1, 0, 2], alpha=0.3, zorder=1)
        ghost_imgs.append(im)
    # Create image artists for each car
    car_imgs = []
    x_positions = [0, 0, 0, 0]  # x positions of cars
    for i in range(4):
        im = ax.imshow(car_icon, extent=[cars_z[0,i]-1, cars_z[0,i]+1, 0, 2], alpha=0, zorder=5)
        car_imgs.append(im)
    # --- Moving trees ---
    tree_path = os.path.join(script_dir, 'tree.png')
    tree_icon = mpimg.imread(tree_path)
    num_trees = 5
    tree_y = np.ones(num_trees)
    tree_x = np.linspace(0, ax.get_xlim()[1], num_trees)
    tree_imgs = []
    for i in range(num_trees):
        im = ax.imshow(tree_icon,
                    extent=[tree_x[i]-0.5, tree_x[i]+0.5, 0, 0], alpha = 0,
                    zorder=0)
        tree_imgs.append(im)

    tree_speed = 10 * dt  # units per frame
    # --- Animation function ---
    def animate(k):
        # update car positions
        for i in range(4):
            car_imgs[i].set_extent(extent=[cars_z[k,i]-1, cars_z[k,i]+1, 0, 2])
            if k > 0:
                car_imgs[i].set_alpha(1)
        # update trees
        for i in range(num_trees):
            if tree_imgs[i].get_extent()[0] + tree_speed > ax.get_xlim()[1]:
                x0 = (tree_imgs[i].get_extent()[0] + tree_speed) - ax.get_xlim()[1] + ax.get_xlim()[0]
            else:
                x0 = tree_imgs[i].get_extent()[0] + tree_speed
            x1 = x0 + 1

            tree_imgs[i].set_extent([x0, x1, 0, 3])
            if k > 0:
                tree_imgs[i].set_alpha(0.5)
        return car_imgs + tree_imgs
    
    # def animate(k):
    #     for i in range(4):
    #         car_imgs[i].set_extent(extent=[cars_z[k,i]-1, cars_z[k,i]+1, 0, 2])
    #     return car_imgs

    anim = FuncAnimation(fig, animate, frames = N, interval= dt*100, blit=True)

    plt.show()

    print("Optimal structured controller Kopt:",Kopt.D)
