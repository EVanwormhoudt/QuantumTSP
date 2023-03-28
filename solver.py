import dimod
import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel

from dwave.system import DWaveSampler, EmbeddingComposite
from dimod.reference.samplers import ExactSolver
from utils import load_tsp,simulated_annealing,pretty_print, verify_solution,cost_function

LAMBDA = 200
NUM_READS = 1000



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
            if i == j:
                continue
            for p in range(n):
                bqm.add_quadratic('x_{}_{}'.format(i, p), 'x_{}_{}'.format(j, (p+1)%n), matrix[i, j])

    for i in range(n):
        for p in range(n):
            for q in range(p+1, n):
                bqm.add_quadratic('x_{}_{}'.format(i, p), 'x_{}_{}'.format(i, q), 2*LAMBDA)
            bqm.add_linear('x_{}_{}'.format(i, p), -LAMBDA)
            bqm.offset += LAMBDA

    for p in range(n):
        for i in range(n):
            for j in range(i+1, n):
                bqm.add_quadratic('x_{}_{}'.format(i, p), 'x_{}_{}'.format(j, p), 2*LAMBDA)
            bqm.add_linear('x_{}_{}'.format(i, p), -LAMBDA)
            bqm.offset += LAMBDA

    return bqm



def solve_tsp(matrix):
    sampler = EmbeddingComposite(DWaveSampler("Advantage2_prototype1.1"))
    bqm = create_bqm(matrix)
    sampleset = sampler.sample(bqm, num_reads=NUM_READS)
    return sampleset

def sort_sampleset(matrix,sampleset):
    
    verified_samples,verified_energies = [],[]
    for i, sample in enumerate(sampleset):
        if verify_solution(sample, matrix):
            verified_samples.append(sample)
            pretty_print(sample,matrix)

    

def main():
    matrix = load_tsp('data/tsp_10.tsp')
    simulated_annealing(matrix,100, 0.01, 0.9, 1000)
    sampleset = solve_tsp(matrix)
    print(sampleset.first.sample)
    sort_sampleset(matrix,sampleset)





main()

