import numpy as np
import control as ctrl

def ivec_DTstatespace(sys, ny, nu):
    """
    Inverse vectorization of a discrete-time state-space system.
    
    Parameters
    ----------
    sys : control.StateSpace or tuple
        Input discrete-time state-space system (A,B,C,D) or control.ss object
    ny : int
        Number of outputs per channel
    nu : int
        Number of inputs / channels
    
    Returns
    -------
    sys_out : control.StateSpace
        Transformed discrete-time state-space system
    """
    # Convert to StateSpace if tuple is provided
    if not isinstance(sys, ctrl.StateSpace):
        sys = ctrl.ss(*sys)
    
    # Extract matrices
    A = np.array(sys.A)
    B = np.array(sys.B)
    C = np.array(sys.C)
    D = np.array(sys.D)
    dt = sys.dt
    
    # Kronecker products for repeated inputs
    Ares = np.kron(np.eye(nu), A)
    Bres = np.kron(np.eye(nu), B)
    Dres = D.reshape(ny, nu)
    
    nx = A.shape[1]
    Cres = np.zeros((ny, nu*nx))
    
    # Fill Cres column blocks
    for jj in range(nu):
        row_start = ny * jj
        row_end = ny * (jj+1)
        col_start = nx * jj
        col_end = nx * (jj+1)
        Cres[:, col_start:col_end] = C[row_start:row_end, :]
    
    # Construct new state-space system
    sys_out = ctrl.ss(Ares, Bres, Cres, Dres, dt)
    return sys_out