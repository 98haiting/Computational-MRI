'''
Author: Bruno Riemenschneider
Email: bruno.riemenschneider@gmail.com
'''

import numpy as np
from scipy.sparse import coo_matrix


# Can you modify this grid() function to be able to take 
def grid(d, k, n):
    """Grid non-cartesian kspace data to a cartesion grid
    Keyword Arguments:
      d - 2D numpy array, non-cartesian kspace
      k - 2D numpy array, kspace trajectory, scaled -0.5 to 0.5
      n - int, grid size
    Returns:
      2D numpy array (n, n)
    """

    if type(n) is not int:
      n = int(n)
      
    # convert the kspace location to matrix index scale/coordinate
    nx = (n - 1) / 2 + (n - 1) * k.real
    ny = (n - 1) / 2 + (n - 1) * k.imag

    m = np.zeros((n, n), dtype=d.dtype)

    # loop over locations in kernel
    for lx in [-1, 0, 1]:
        for ly in [-1, 0, 1]:
            # find the nearest cartesian kspace point(in m)
            nxt = np.round(nx + lx)
            nyt = np.round(ny + ly)

            # calculate the weighting for triangular kernel
            kwx = 1 - np.abs(nx - nxt)
            kwx[kwx < 0] = 0
            kwy = 1 - np.abs(ny - nyt)
            kwy[kwy < 0] = 0

            # adjust for the sample index on the edge of the matrix
            nxt[nxt < 0] = 0
            nyt[nyt < 0] = 0
            nxt[nxt > n - 1] = n - 1
            nyt[nyt > n - 1] = n - 1

            # gridding
            temp = d * kwx * kwy
            m = m + coo_matrix((temp.flatten(), (nxt.flatten(), nyt.flatten())), shape=(n, n)).toarray()

    m[:, 0] = 0
    m[:, -1] = 0
    m[0, :] = 0
    m[-1, :] = 0
    return m
