import os

import dimod

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel

from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite
from dimod.reference.samplers import ExactSolver
from minorminer import find_embedding
from  dwave import inspector
import time


from utils import load_tsp,simulated_annealing,pretty_print, verify_solution_tsp,cost_function,plot_solution
from tsp_file_generation import generate_tsp_file

LAMBDA = 75
NUM_READS = 500
NUM_CITY = 8
TOPOLOGY = "pegasus"
ANNEALING_TIME = 100




def create_bqm_TSP(matrix):
    bqm = BinaryQuadraticModel.empty(dimod.BINARY)
    n = len(matrix)
    # add the quadratic terms for the distance between cities
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for p in range(n):
                bqm.add_quadratic('x_{}_{}'.format(i, p),
                                  'x_{}_{}'.format(j, (p+1)%n), matrix[i, j])
    # add the constraint that there is only one city visited at each position
    for i in range(n):
        for p in range(n):
            for q in range(p+1, n):
                bqm.add_quadratic('x_{}_{}'.format(i, p),
                                  'x_{}_{}'.format(i, q), 2*LAMBDA)
            bqm.add_linear('x_{}_{}'.format(i, p), -LAMBDA)
        bqm.offset += LAMBDA

    # add the constraint that each city is visited exactly once
    for p in range(n):
        for i in range(n):
            for j in range(i+1, n):
                bqm.add_quadratic('x_{}_{}'.format(i, p),
                                  'x_{}_{}'.format(j, p), 2*LAMBDA)
            bqm.add_linear('x_{}_{}'.format(i, p), -LAMBDA)
        bqm.offset += LAMBDA

    return bqm






def solve_tsp(matrix):
    sampler = LazyFixedEmbeddingComposite(DWaveSampler(solver={'topology__type': TOPOLOGY}))
    bqm = create_bqm_TSP(matrix)

    sampleset = sampler.sample(bqm, num_reads=NUM_READS, annealing_time=ANNEALING_TIME, label='TSP')
    #print average energy of samples
    print("Average energy: ", np.mean(sampleset.record.energy))

    embedding = sampleset.info['embedding_context']['embedding']

    print(f"Number of logical variables: {len(embedding.keys())}")
    print(f"Number of physical qubits used in embedding: {sum(len(chain) for chain in embedding.values())}")

    inspector.show(sampleset)
    return sampleset

def print_sampleset(matrix,sampleset,best):

    verified_samples = []
    for i, sample in enumerate(sampleset):
        if verify_solution_tsp(sample, matrix):
            solution = pretty_print(sample,matrix)
            verified_samples.append(solution)

    print("Number of solutions found: ", len(verified_samples))
    costs = [cost_function(x,matrix) for x in verified_samples]
    if len(verified_samples) == 0:
        print("No solution found")
        return

    print("Best solution: ", verified_samples[np.argmin(costs)])
    print("Cost: ", np.min(costs))
    plot_solution( verified_samples[np.argmin(costs)],matrix)



def main():
    # check if the file exists
    if not os.path.isfile('data/tsp_'+str(NUM_CITY)+'.tsp'):
        generate_tsp_file(NUM_CITY)

    matrix = load_tsp('data/tsp_'+str(NUM_CITY)+'.tsp')
    start = time.time()
    print("Simulated Annealing")
    best = simulated_annealing(matrix,100, 0.01, 0.9, 1000)
    print("Time: ", time.time()-start)
    print("-----------------------------")
    print("Quantum Annealing")
    start = time.time()
    sampleset = solve_tsp(matrix)
    print_sampleset(matrix,sampleset,best)
    print("Time: ", time.time()-start)



main()
