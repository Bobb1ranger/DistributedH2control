import numpy as np
from scipy.linalg import qr
import control as ctrl
def controllable_subspace(A, B, tol=1e-6):
    """
    Computes a numerically stable basis for the controllable subspace of (A, B).

    Parameters
    ----------
    A : ndarray (n x n)
        System matrix
    B : ndarray (n x m)
        Input matrix
    tol : float, optional
        Numerical tolerance for rank determination (default: 1e-10)

    Returns
    -------
    Uc : ndarray (n x r)
        Orthonormal basis for the controllable subspace
    Ac : ndarray (r x r)
        Reduced A matrix
    Bc : ndarray (r x m)
        Reduced B matrix
    """

    n = A.shape[0]

    # Start with span(B)
    U = B.copy()

    # QR factorization to get an orthonormal basis
    Q, _ = qr(U, mode='economic')
    U = Q

    old_rank = 0

    # Iteratively expand until rank stabilizes
    while True:
        U_new = np.hstack((U, A @ U))

        # QR with column pivoting
        Q, R, _ = qr(U_new, mode='economic', pivoting=True)

        # Numerical rank detection
        if R.size == 0:
            new_rank = 0
        else:
            new_rank = np.sum(np.abs(np.diag(R)) > tol * np.abs(R[0, 0]))

        if new_rank == old_rank:
            break

        U = Q[:, :new_rank]
        old_rank = new_rank

    # Project A and B onto the controllable subspace
    Ac = U.T @ A @ U
    Bc = U.T @ B

    Uc = U
    return Uc, Ac, Bc

def minimal(sys, tol=1e-10):
    """
    Compute a minimal realization of a discrete-time state-space system
    and return the projection matrix.
    
    Parameters
    ----------
    sys : control.StateSpace
        Input SS system
    tol : float
        Tolerance for rank determination in controllable_subspace
    
    Returns
    -------
    sysm : control.StateSpace
        Minimal realization (controllable + observable)
    T : ndarray
        Projection matrix from original states to minimal states
    """
    if not isinstance(sys, ctrl.StateSpace):
        sys = ctrl.ss(*sys)
    
    A = np.array(sys.A)
    B = np.array(sys.B)
    C = np.array(sys.C)
    D = np.array(sys.D)
    dt = sys.dt
    
    # Step 1: Remove uncontrollable states
    Uc, Ac, Bc = controllable_subspace(A, B, tol)
    A1 = Ac
    B1 = Bc
    C1 = C @ Uc
    T1 = Uc  # Projection for controllable subspace
    
    # Step 2: Remove unobservable states
    Uo, Ao_temp, Co_temp = controllable_subspace(A1.T, C1.T, tol)
    Ao = Ao_temp.T
    Bo = (B1.T @ Uo).T
    Co = C1 @ Uo
    T2 = Uo  # Projection for observable subspace
    
    # Full projection from original state to minimal state
    T = T1 @ T2
    
    sysm = ctrl.ss(Ao, Bo, Co, D, dt)
    return sysm, T