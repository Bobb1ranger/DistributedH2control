import numpy as np
import control as ct
from vectorize import mat_vec, orth_basis
from scipy.linalg import solve_discrete_are, kron
from h2_riccati import h2_state_feedback_gain, h2_kalman_filter_gain
from Controllable_subspace import controllable_subspace
import ivec_DTss
import minreal
from Controllable_subspace import minimal
from feedback import lft
from control import norm

import time
class DiscreteGeneralizedPlant:
    """
    Class representing a discrete-time generalized plant:
        x[k+1] = A x[k] + B1 w[k] + B2 u[k]
        z[k]   = C1 x[k] + D11 w[k] + D12 u[k]
        y[k]   = C2 x[k] + D21 w[k] + D22 u[k]
    """
    def __init__(self, A, B1, B2, C1, C2, D11, D12, D21, D22, dt=1.0):
        self.A = np.array(A)
        self.B1 = np.array(B1)
        self.B2 = np.array(B2)
        self.C1 = np.array(C1)
        self.C2 = np.array(C2)
        self.D11 = np.array(D11)
        self.D12 = np.array(D12)
        self.D21 = np.array(D21)
        self.D22 = np.array(D22)
        self.dt = dt

        self.nx = self.A.shape[0]
        self.nw = self.B1.shape[1]
        self.nu = self.B2.shape[1]
        self.ny = self.C2.shape[0]
        self.nz = self.C1.shape[0]



def structured_output_feedback_h2(plant, rel_order):
    """
    Structured H2 output-feedback synthesis.
    
    Inputs:
        plant: DiscreteGeneralizedPlant object
        rel_order: ny x nu matrix
            - entry >=0 : relative order of that controller entry
            - entry == -1 : structurally zero entry
    Outputs:
        K_struct: optimal output-feedback controller with given relative order
    """
    # Extract plant matrices
    A = plant.A
    B2 = plant.B2
    C2 = plant.C2
    D22 = plant.D22
    B1 = plant.B1
    C1 = plant.C1
    D11 = plant.D11
    D12 = plant.D12
    D21 = plant.D21
    dt = plant.dt

    nx, nu, ny = plant.nx, plant.nu, plant.ny
    nw, nz = plant.nw, plant.nz

    # 1. Weighting matrices



    # 2. Full-state feedback and estimator gains placeholders
    F2, X2 = h2_state_feedback_gain(A, B2, C1, D12)  # state-feedback gain
    L2, Y2 = h2_kalman_filter_gain(A, B1, C2, D21)
    Rb = D12.T @ D12 + B2.T @ X2 @ B2
    Rc = D21 @ D21.T + C2 @ Y2 @ C2.T
    term1 = B2.T @ X2 @ A @ Y2 @ C2.T
    term2 = D12.T @ C1 @ Y2 @ C2.T
    term3 = B2.T @ X2 @ B1 @ D21.T
    term4 = D12.T @ D11 @ D21.T

    # Sum all terms
    numerator = term1 + term2 + term3 + term4

    # Multiply by inverses of Rb and Rc
    # Make sure to use np.linalg.inv for matrix inversion
    L0 = - np.linalg.inv(Rb) @ numerator @ np.linalg.inv(Rc)
        

    

    # 3. Construct augmentde system for Riccati recursion
    R = kron(Rc,Rb)

    # Construct augmented matrices for recursion
    Av = np.block([
        [kron(np.eye(ny), A), np.zeros((nx*ny, nx*nu))],
        [-kron(C2.T, F2), kron(A.T, np.eye(nu))]
    ])
    Bv = np.vstack([
        kron(np.eye(ny), B2),
        kron(C2.T, np.eye(nu))
    ])
    Cv = -np.hstack([
        kron(np.eye(ny), F2),
        kron(L2.T, np.eye(nu))
    ])
    Dv = np.eye(nu*ny)
    b0 = np.vstack([np.zeros((nx*ny,1)), F2.flatten(order='F').reshape(-1,1)])
    # Augment with delays using relative-order matrix
    m = np.max(rel_order[rel_order>=0]) if np.any(rel_order>=0) else 0
    Br_list = [None]*(m+1)
    Dr_list = [None]*(m+1)
    Xred_list = [None]*(m+1)
    Kred_list = [None]*(m+1)

    # Build delay mask for orth_basis
    delay_mask = rel_order >= 0
    Em = orth_basis(delay_mask)

    # Concatenate [Bv*Em, b0]
    B_aug = np.hstack((Bv @ Em, b0))

    # Controllable subspace reduction
    tol = 1e-4
    TR, Ar, _ = controllable_subspace(Av, B_aug, tol)
    # Left projection
    TL = TR.T

    # Reduced C matrix
    Cr = Cv @ TR

    # Reduced B matrices
    Brm = TL @ Bv @ Em
    Dm = Em


    Q = Cr.T @ R @ Cr
    Rm = Dm.T @ R @ Dm
    S = Cr.T @ R @ Dm

    Rinv = np.linalg.inv(Rm)
    Av_tilde = Ar - Brm @ Rinv @ S.T
    Q_tilde = Q - S @ Rinv @ S.T

    # Terminal Riccati
    Xred_list[m] = solve_discrete_are(Av_tilde, Brm, Q_tilde, Rm)
    Kred_list[m] = - np.linalg.inv(Brm.T @ Xred_list[m] @ Brm + Rm) @ ( Ar.T @ Xred_list[m] @ Brm + S).T  # simplified

      
    # Backward recursion
    for ii in reversed(range(m)):
        delay_mask = (rel_order >= 0) & (rel_order <= ii)
        Eii = orth_basis(delay_mask)
        Br_list[ii] = TL @ Bv @ Eii
        Dr_list[ii] = Eii
        V = Ar.T @ Xred_list[ii+1] @ Br_list[ii] + Cr.T @ R @ Dr_list[ii]
        Kred_list[ii] = -np.linalg.inv(Br_list[ii].T @ Xred_list[ii+1] @ Br_list[ii] + Dr_list[ii].T @ R @ Dr_list[ii]) @ V.T
        Xred_list[ii] = Ar.T @ Xred_list[ii+1] @ Ar + Cr.T @ R @ Cr + V @ Kred_list[ii]

    L0_vec = mat_vec(L0)

    # Numerator
    num = Br_list[0].T @ Xred_list[1] @ TL @ b0 + Dr_list[0].T @ R @ L0_vec

    # Denominator
    den = Br_list[0].T @ Xred_list[1] @ Br_list[0] + Dr_list[0].T @ R @ Dr_list[0]

    u0opt = -np.linalg.solve(den, num)

    # u0opt is computed earlier
    q0 = Dr_list[0] @ u0opt

    # Initialize lists
    zeta = [None] * m
    q = [None] * (m-1)

    # First state
    zeta[0] = Br_list[0] @ u0opt + TL @ b0

    # Forward recursion to compute zeta and q
    for ii in range(1, m):
        zeta[ii] = (Ar + Br_list[ii] @ Kred_list[ii]) @ zeta[ii-1]
        q[ii-1] = (Cr + Dr_list[ii] @ Kred_list[ii]) @ zeta[ii-1]

    # Tail system (like minreal in MATLAB)
    # Here, build a discrete-time StateSpace for the tail
    # qtail = ss(Ar+Brm*Kred[-1], zeta[m-1], Cr+Dm*Kred[-1], 0, dt)
    # In Python:

    A_tail = Ar + Brm @ Kred_list[-1]
    B_tail = zeta[m-1].reshape(-1,1)   # zeta[m] is column vector
    C_tail = Cr + Dm @ Kred_list[-1]
    D_tail = np.zeros((C_tail.shape[0], B_tail.shape[1]))

    qtail = ct.ss(A_tail, B_tail, C_tail, D_tail, dt)

    if m > 1:
        A_delay = np.zeros((m-1, m-1))
        if m > 2:
            A_delay[0:(m-2), 1:(m-1)] = np.eye(m-2)
        B_delay = np.zeros((m-1,1))
        B_delay[-1,0] = 1.0
        C_delay = np.eye(m-1)[:,0].reshape(1,-1)  # output the last delayed input
        D_delay = np.zeros((1,1))
        delay_sys = ct.ss(A_delay, B_delay, C_delay, D_delay, dt)
        # Series connection: tail goes through delay
        qtail_delayed = ct.series(delay_sys, qtail)
    else:
        qtail_delayed = qtail
    # Build vecQ_fir (FIR part)
    if m > 1:
        # FIR state-space (shift register)
        A_fir = np.zeros((m-1, m-1))
        if m > 2:
            A_fir[0:(m-2), 1:(m-1)] = np.eye(m-2)
        B_fir = np.zeros((m-1, 1))
        B_fir[-1, 0] = 1.0

        # C_fir concatenates q[m-1] down to q[0]
        C_fir = np.hstack([q[ii] for ii in reversed(range(m-1))])
        D_fir = q0

    else:
        # If m==1, FIR is just D = q0
        A_fir = np.zeros((0,0))
        B_fir = np.zeros((0,1))
        C_fir = np.zeros((q0.shape[0],0))
        D_fir = q0

    # Combine FIR and tail as needed
    # In Python, you can leave them separate or implement series connection
    # For example, using control library:


    vecQ_fir = ct.ss(A_fir, B_fir, C_fir, D_fir, dt)
    vecQ_ss = ct.parallel(vecQ_fir, qtail_delayed)
    tol = 1e-6
    Q, _ = minimal(ivec_DTss.ivec_DTstatespace(vecQ_ss, nu, ny),tol)

    # Compute Jy matrices
    AJy = A + B2 @ F2 + L2 @ C2 + L2 @ D22 @ F2
    BJy = np.hstack([-L2, - B2 - L2 @ D22])
    CJy = np.vstack([F2, -C2 - D22 @ F2])
    DJy = np.block([[np.zeros((nu, ny)), -np.eye(nu)],
                    [np.eye(ny), D22]])

    # Construct Jy as state-space
    Jy = ct.ss(AJy, BJy, CJy, DJy, dt)

    # Compute LFT: KH2dis = minimal2(lft(Jy, -Q), tol)
    print(AJy,BJy,CJy,DJy)

    KH2dis, _ = minimal(lft(Jy, Q), tol)
    # KH2dis = lft(Jy, Q)
    # Compute H2 norm (or 2-norm) of closed-loop system


    return KH2dis



# ---------------- Example Usage ----------------
if __name__ == "__main__":


    # Define a toy 2x2 plant
    A = np.array([
        [0, 3.3, 0, -0.4, 0, 0],
        [-0.1, 0.7, 0.1, -0.5, 0, 0],
        [0, -0.7, 0.4, -0.6, 0, 0],
        [-0.4, 1.7, 0.6, -1.6, 0, 0],
        [-0.1, -1.4, 0, 0, -0.2, 0.7],
        [-2.0, -0.8, 0, 0, 0.3, 1]
    ], dtype=float)

    B1 = np.array([
        [0.6, -0.6, -0.5],
        [0.2, 0.6, 0.4],
        [1.0, -1.0, -2.2],
        [0.0, -1.0, 0.0],
        [0.3, 0.5, 0.2],
        [0.0, -0.5, 0.3]
    ], dtype=float)

    C1 = np.array([
        [0.1, 0.4, 0.1, 0.4, 0.1, 0.4],
        [0.3, 0.0, 0.5, 1.0, 0.6, 2.0],
        [-1.0, -0.1, 0.2, -0.5, 0.3, 1.0]
    ], dtype=float)

    B2 = np.array([
        [0.8, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -0.4]
    ], dtype=float)

    C2 = np.array([
        [0.2, 1.0, 0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.2, -1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.2, -1.0]
    ], dtype=float)

    D11 = np.zeros((3, 3), dtype=float)
    D12 = np.eye(3, dtype=float)
    D21 = np.eye(3, dtype=float)
    D22 = np.zeros((3, 3), dtype=float)

    start_time = time.perf_counter()

    # --- Your code goes here ---
    plant = DiscreteGeneralizedPlant(A, B1, B2, C1, C2, D11, D12, D21, D22, dt=1)

    # plant = DiscreteGeneralizedPlant(A,B1,B2,C1,C2,D11,D12,D21,D22,dt=0.1)
    rel_order = np.array([[0, 1, -1],
                          [1, 0, -1],
                          [1, 2, 0]])  # -1 means zero entry
    # rel_order = np.array([[0, -1],[1,0]])  # -1 means zero entry

    Kopt = structured_output_feedback_h2(plant, rel_order)
    # --------------------------




    print("Structured H2 output-feedback gain:\n", norm(ct.feedback(Kopt,- ct.ss(A,B2,C2,D22,dt = 1))))

    end_time = time.perf_counter()

    # Calculate and print the runtime
    runtime = end_time - start_time
    print(f"Execution time: {runtime:.4f} seconds")