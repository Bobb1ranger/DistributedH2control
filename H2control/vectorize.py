import numpy as np

def mat_vec(M):
    return M.flatten(order='F').reshape(-1, 1)

def orth_basis(spm):
    """
    Python equivalent of MATLAB orthBasis(spm).

    Parameters
    ----------
    spm : array-like
        Boolean or numeric mask. Nonzero entries define the support.

    Returns
    -------
    B : ndarray
        Selection matrix whose columns are standard basis vectors
        corresponding to nonzero entries of spm.
    """
    spm = np.asarray(spm)
    S = spm.size
    spm_flat = spm.flatten(order='F')
    supp = np.nonzero(spm_flat)[0]

    E = np.zeros((S, len(supp)))
    for i, idx in enumerate(supp):
        E[idx, i] = 1.0

    return E