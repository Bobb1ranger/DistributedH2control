import numpy as np
import control as ctrl

def stair(A, B, tol):
    """
    Stair algorithm to detect uncontrollable or unobservable states.
    
    Parameters
    ----------
    A : ndarray (ns x ns)
    B : ndarray
    tol : float
    
    Returns
    -------
    T : ndarray (ns x ns) - transformation matrix
    nu : int - number of states removed
    """
    ns = A.shape[0]
    bx = B.copy()
    ax = A.copy()
    T = np.eye(ns)
    
    r = ns
    while r > 0:
        rbx, cbx = bx.shape
        # SVD
        U, S, Vh = np.linalg.svd(bx)
        s_diag = S[:min(rbx, cbx)]
        r = np.sum(s_diag > tol)
        
        if r == 0 or r == rbx:
            break
        
        Ttemp = np.eye(ns)
        Ttemp[-rbx:, -rbx:] = U
        T = T @ Ttemp
        ax = U.T @ ax @ U
        if rbx - r > 0:
            bx = ax[r:rbx, :r]
            ax = ax[r:rbx, r:rbx]
        else:
            bx = np.zeros((0, r))
            ax = np.zeros((0, 0))
    
    nu = rbx if r == 0 else 0
    return T, nu


def minimal2(sys, tol= 1e-6):
    """
    Remove uncontrollable and unobservable states from a discrete-time state-space system.
    
    Parameters
    ----------
    sys : control.StateSpace
        Input SS system
    tol : float, optional
        Tolerance for rank determination (default: ns^2 * ||A||_1 * eps)
    
    Returns
    -------
    sysm : control.StateSpace
        Minimal realization of the system
    T : ndarray
        Transformation matrix applied
    """
    if not isinstance(sys, ctrl.StateSpace):
        sys = ctrl.ss(*sys)
    
    A = np.array(sys.A)
    B = np.array(sys.B)
    C = np.array(sys.C)
    D = np.array(sys.D)
    Ts = sys.dt
    
    ns, ni = B.shape
    no, _ = C.shape
    
    if tol is None:
        tol = ns**2 * np.linalg.norm(A, 1) * np.finfo(float).eps
    
    T1 = np.eye(ns)
    T1, nuc = stair(A, B, tol)
    if nuc > 0:
        T1_ext = np.block([
            [np.eye(ns - nuc), np.zeros((ns - nuc, nuc))],
            [np.zeros((nuc, ns - nuc)), np.eye(nuc)]
        ])
        T1 = T1 @ T1_ext
        A = T1.T @ A @ T1
        B = T1.T @ B
        C = C @ T1
    
    T2 = np.eye(ns)
    ns2 = A.shape[0]
    T2, nuo = stair(A.T, C.T, tol)
    if nuo > 0:
        T2_ext = np.block([
            [np.eye(ns2 - nuo), np.zeros((ns2 - nuo, nuo))],
            [np.zeros((nuo, ns2 - nuo)), np.eye(nuo)]
        ])
        T2 = T2 @ T2_ext
        A = T2.T @ A @ T2
        B = T2.T @ B
        C = C @ T2
    
    T = T1 @ T2
    nsr = nuc + nuo
    print(f"{nsr} states removed non minimal")
    
    if 0 < nsr < ns:
        sysm = ctrl.ss(A, B, C, D, Ts)
    elif nsr == ns:
        sysm = ctrl.ss(np.zeros((0,0)), np.zeros((0,B.shape[1])), np.zeros((C.shape[0],0)), D, Ts)
    else:
        sysm = sys
    
    return sysm, T
