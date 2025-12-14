
import numpy as np
import control as ct

def simulate_discrete_ss(sys, u_seq, x0):
    """
    Simulate a discrete-time LTI system using its state-space object.

    sys: control.StateSpace (discrete-time)
    u_seq: shape (N, m) input sequence
    x0: shape (n,) initial state
    """
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    dt = sys.dt  # optional if you want timestamps

    N = u_seq.shape[0]
    n = A.shape[0]
    p = C.shape[0]

    x = np.zeros((N+1, n))
    y = np.zeros((N, p))

    x[0] = x0

    for k in range(N):
        y[k] = C @ x[k] + D @ u_seq[k]
        x[k+1] = A @ x[k] + B @ u_seq[k]


    return x, y