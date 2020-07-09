import numpy as np
import torch

import utils


def test_normsq():
    x = 2 * np.ones((3, 3))
    assert utils.normsq(x) == 36

    x = -torch.ones((2, 3))
    assert utils.normsq(x) == 6


def test_mnls():

    A = np.array(
        [[1, 2, 0],
         [1, 2, 0]]
    )

    b = np.array([1, 0])

    np_solution = np.linalg.pinv(A) @ b
    iterative_solution = utils.mnls(lambda x: A@x, lambda x: A.T@x,
                                    b, np.zeros(A.shape[1]))

    assert np.allclose(np_solution, iterative_solution)
