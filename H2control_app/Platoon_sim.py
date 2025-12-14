import numpy as np
import control as ct

from scipy.linalg import kron

from H2control.DistributedH2controlTF import structured_output_feedback_h2
import H2control.DistributedH2controlTF as dh2


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
    tau = 0.15

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
    dt = 1/64
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
        C2[ii, ii*3 - 3] = -1

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
