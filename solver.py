import dimod
import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel

from dwave.system import DWaveSampler, EmbeddingComposite
from dimod.reference.samplers import ExactSolver
from utils import load_tsp,simulated_annealing,pretty_print, verify_solution,cost_function,plot_solution

LAMBDA = 200
NUM_READS = 1000



def create_bqm(matrix):
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
    sampler = EmbeddingComposite(DWaveSampler())
    bqm = create_bqm(matrix)
    sampleset = sampler.sample(bqm, num_reads=NUM_READS)
    return sampleset

def print_sampleset(matrix,sampleset,best):
    
    verified_samples = []
    for i, sample in enumerate(sampleset):
        if verify_solution(sample, matrix):
            solution = pretty_print(sample,matrix)
            verified_samples.append(solution)

    costs = [cost_function(x,matrix) for x in verified_samples]
    print("Best solution: ", verified_samples[np.argmin(costs)])
    print("Cost: ", np.min(costs))
    plot_solution( verified_samples[np.argmin(costs)],matrix)



def main():
    matrix = load_tsp('data/tsp_10.tsp')
    best = simulated_annealing(matrix,100, 0.01, 0.9, 1000)
    sampleset = solve_tsp(matrix)
    print_sampleset(matrix,sampleset,best)







main()

