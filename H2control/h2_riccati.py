import numpy as np
from scipy.linalg import solve_discrete_are

def h2_state_feedback_gain(A, B2, C1, D12):
    """
    Compute H2-optimal state-feedback gain F using LQR transformation.
    """

    Q = C1.T @ C1
    R = D12.T @ D12
    S = C1.T @ D12

    # LQR transformation
    Rinv = np.linalg.inv(R)
    A_tilde = A - B2 @ Rinv @ S.T
    Q_tilde = Q - S @ Rinv @ S.T

    # Solve standard DARE
    X = solve_discrete_are(A_tilde, B2, Q_tilde, R)

    # Optimal H2 gain
    F = -np.linalg.inv(R + B2.T @ X @ B2) @ (B2.T @ X @ A + S.T)

    return F, X

def h2_kalman_filter_gain(A, B1, C2, D21):
    """
    Compute infinite-horizon H2-optimal Kalman filter gain L
    using LQR transformation (dual Riccati equation).

    System:
        x_{k+1} = A x_k + B1 w_k
        y_k     = C2 x_k + D21 w_k
    """

    # Estimation Riccati weights
    W = B1 @ B1.T
    V = D21 @ D21.T
    S = B1 @ D21.T

    # LQR (dual) transformation
    Vinv = np.linalg.inv(V)
    A_tilde = A - S @ Vinv @ C2
    W_tilde = W - S @ Vinv @ S.T

    # Solve dual discrete-time Riccati equation
    Y = solve_discrete_are(A_tilde.T, C2.T, W_tilde, V)

    # Optimal Kalman gain
    L = -(A @ Y @ C2.T + S) @ np.linalg.inv(C2 @ Y @ C2.T + V)

    return L, Y

if __name__ == "__main__":

    # ----- Continuous-time system -----
    A = np.array([[-0.1, 0.0],
                   [-1.0, 0.0]])
    B2 = np.array([[1],
                   [0]])
    C1 = np.array([[0.0, 1.0]])
    D12 = np.array([[1]])

    F, X = h2_state_feedback_gain(A, B2, C1, D12)

    print("H2-optimal state-feedback gain F:")
    print(F)
    print("Solution to DARE X:")
    print(X)

    L, Y = h2_state_feedback_gain(A, C1.T, B2.T, D12.T)
    print("H2-optimal Kalman filter gain L:")
    print(L)
    print("Solution to dual DARE Y:")
    print(Y)