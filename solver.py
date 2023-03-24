import dimod
import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dwave.system import LeapHybridSampler
from utils import load_tsp,simulated_annealing

def create_bqm(matrix):
    """Create a BQM for the TSP problem.

    Args:
        matrix (numpy.ndarray): The distance matrix.

    Returns:
        dimod.BinaryQuadraticModel: The BQM.

    """

    bqm = BinaryQuadraticModel.empty(dimod.BINARY)
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            for p in range(n):
                bqm.add_quadratic('x_{}_{}'.format(i, p), 'x_{}_{}'.format(j, (p+1)%n), matrix[i, j])

    for i in range(n):
        #bqm.add_linear('x_{}_{}'.format(i, i), 0)
        continue

    return bqm

def solve_tsp(matrix):
    sampler = LeapHybridSampler()
    bqm = create_bqm(matrix)
    sampleset = sampler.sample(bqm, label='TSP')
    return sampleset


def main():
    matrix = load_tsp('data/tsp_100.tsp')
    sampleset = solve_tsp(matrix)
    print(sampleset)




