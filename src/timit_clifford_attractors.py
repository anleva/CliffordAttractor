import timeit

BEST_OF = 3
REPEAT = 3

statement = """
h = sample_histogram(a=-1.4, b=1.6, c=1.0, d=0.7, h=None, dim=1000, samples=1e6, randomize=1000)
"""


numpy_numba_setup = """
from clifford_attractor_numpy_numba import sample_histogram
"""
print('Execution time NumPy with Numba: {}'.format(
    min(timeit.Timer(statement, numpy_numba_setup).repeat(BEST_OF, REPEAT))))


math_math_numba_setup = """
from clifford_attractor_math_numba import sample_histogram
"""
print('Execution time math with Numba: {}'.format(
    min(timeit.Timer(statement, math_math_numba_setup).repeat(BEST_OF, REPEAT))))


math_setup = """
from clifford_attractor_math import sample_histogram
"""
print('Execution time pure math: {}'.format(
    min(timeit.Timer(statement, math_setup).repeat(BEST_OF, REPEAT))))


numpy_setup = """
from clifford_attractor_numpy import sample_histogram
"""
print('Execution time pure NumPy: {}'.format(
    min(timeit.Timer(statement, numpy_setup).repeat(BEST_OF, REPEAT))))

