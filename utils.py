import numpy as np

def normsq(x):
    return (x * x).sum()

def mnls(A, b, x0=None, num_steps=10, tol=1e-14):
    """
    Returns the solution to ``argmin_x || Ax - b ||_2`` closest to ``x_0``.
    By default, ``x_0`` is zero and therefore returns the minimum norm
    solution.

    Parameters
    ----------
    A : LinearOperator, shape (m, n)?
    b : vector
    x0 : vector



    References
    ----------
   Kammerer and Nashed 'On the convergence...'
    """

    if x0 is None:
        x0 = np.zeros(A.shape[1])

    r = p = A.T @ (A @ x0 - b)

    if normsq(r) < tol:
        return x0

    alpha = normsq(r) / normsq(A @ p)
    x = x0 - alpha * p

    # main loop
    for step in range(num_steps):
        r = A.T @ (A @ x - b)

        if normsq(r) < tol:
            return x

        beta = - np.dot(r, A.T @ A @ p) / normsq(A @ p)
        p = r + beta * p

        alpha = np.dot(r, p) / normsq(A @ p)
        x = x + alpha * p

    return x
