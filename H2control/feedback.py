import numpy as np
import control as ct

def lft(P, K):
    """
    Lower Linear Fractional Transformation (LFT)
    
    Given
        P = [P11 P12; P21 P22], K
    returns the closed-loop system: P11 + P12*K*(I - P22*K)^(-1)*P21
    
    Parameters
    ----------
    P : control.StateSpace
        Generalized plant with partitioned input/output
    K : control.StateSpace
        Controller

    Returns
    -------
    sys_cl : control.StateSpace
        Closed-loop system
    """
    # Extract state-space data
    A, B, C, D = P.A, P.B, P.C, P.D
    nx = A.shape[0]
    
    # Partition sizes
    ny, nu = K.B.shape[1], K.C.shape[0]
    nw = B.shape[1] - nu  # assume input partition [w; u]
    nz = C.shape[0] - ny  # assume output partition [z; y]
    
    # Partition P
    P11 = ct.ss(A, B[:, :nw], C[:nz, :], D[:nz, :nw], P.dt)
    P12 = ct.ss(A, B[:, nw:], C[:nz, :], D[:nz, nw:], P.dt)
    P21 = ct.ss(A, B[:, :nw], C[nz:, :], D[nz:, :nw], P.dt)
    P22 = ct.ss(A, B[:, nw:], C[nz:, :], D[nz:, nw:], P.dt)
    
    # Compute LFT: P11 + P12*K*(I - P22*K)^(-1)*P21
    sys_cl = P11 + ct.series(P21, ct.feedback(K, P22, sign = 1), P12)
    return sys_cl