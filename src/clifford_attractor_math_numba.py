import numpy as np
import math
from numba import jit

"""
CLIFFORD ATTRACTORS
Each new point (x_n+1, y_n+1) is determined based on the preceding point (x_n, y_n), and the parameters a, b, c and d. 
x_n+1 = sin(a * y_n) + c * cos(a x_n)
y_n+1 = sin(b * x_n) + d * cos(b y_n)
"""


@jit(nopython=True)
def sample_histogram(a=0.98, b=1.7, c=1.48, d=1.57, h=None, dim=2000, samples=1e9, randomize=1000):
    """
    Samples a number of points for a Clifford Attractor.

    :param a: Attractor Parameter.  Defaults to 0.98.
    :param b: Attractor Parameter.  Defaults to 1.70.
    :param c: Attractor Parameter.  Defaults to 1.48.
    :param d: Attractor Parameter.  Defaults to 1.57.
    :param h: Optional. A sample histogram from a previous sampling. Defaults to None, and initiates a zero-valued array
    :param dim: The output histogram h will have the dimension (dim, dim).
    :param samples: Number of samples generated. Defaults to 1e9.
    :param randomize: The number of samples between each random shock.
    :return: h, updated NumPy array of dimension (dim, dim).
    """

    # Initialize
    x, y = 0, 0
    c_abs, d_abs = np.abs(c), np.abs(d)

    if h is None:
        h = np.zeros((dim, dim))

    # Split sample in batches if too big
    while samples > 1e9:
        h = sample_histogram(a, b, c, d, h, dim, 1e9, randomize)
        samples -= 1e9

    # Run samples
    for i in range(int(samples)):

        # Randomize sometimes to avoid numerical issues
        if i % randomize == 0:
            x += 0.1 * np.random.normal()
            y += 0.1 * np.random.normal()

        # Get new data
        x_ = math.sin(a * y) + c * math.cos(a * x)
        y = math.sin(b * x) + d * math.cos(b * y)
        x = x_

        # Get buckets
        bx = math.floor(dim * (0.03 + 0.94 * (1 + c_abs + x) / (2 + 2 * c_abs)))
        by = math.floor(dim * (0.03 + 0.94 * (1 + d_abs + y) / (2 + 2 * d_abs)))

        # Update histogram
        h[bx, by] += 1

    return h


# h = sample_histogram()


