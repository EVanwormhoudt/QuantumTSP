import random
from copy import deepcopy

import dimod

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel

from dwave.system import DWaveSampler, AutoEmbeddingComposite,EmbeddingComposite
from dimod.reference.samplers import ExactSolver
from matplotlib import pyplot as plt
from minorminer import find_embedding
from  dwave import inspector
import time



from utils import load_gtsp,pretty_print,cost_function

LAMBDA = 150
NUM_READS = 1000
NUM_CITY = 18
TOPOLOGY = "pegasus"
ANNEALING_TIME = 100



def plot_solution(solution,matrix,coords):
    plt.figure(figsize=(10, 10))
    plt.suptitle("Visualisation of the solution")
    # add the cost
    plt.title("Cost: {}".format(cost_function(solution, matrix)))
    plt.scatter(coords[:, 0], coords[:, 1], c='r', s=100)
    plt.plot(coords[solution, 0], coords[solution, 1], c='b', linewidth=2)
    plt.plot([coords[solution[-1], 0], coords[solution[0], 0]], [coords[solution[-1], 1], coords[solution[0], 1]], c='b', linewidth=2)
    plt.show()



def create_bqm_GTSP(matrix, clusters: dict):

    bqm = BinaryQuadraticModel.empty(dimod.BINARY)
    n = len(matrix)
    num_clusters = len(clusters.keys())





    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for p in range(num_clusters):
                bqm.add_quadratic('x_{}_{}'.format(i, p), 'x_{}_{}'.format(j, (p+1)%num_clusters), matrix[i, j])


    for p in range(num_clusters):
        for i in range(n):
            for j in range(i+1, n):
                bqm.add_quadratic('x_{}_{}'.format(i, p), 'x_{}_{}'.format(j, p), 2*LAMBDA)
            bqm.add_linear('x_{}_{}'.format(i, p), -LAMBDA)
        bqm.offset += LAMBDA

    for c in clusters.values():
        for i in c:
            for j in c:
                for p in range(num_clusters):
                    for q in range(p+1,num_clusters):
                        if p == q:
                            continue
                        bqm.add_quadratic('x_{}_{}'.format(i, p), 'x_{}_{}'.format(j, q), 2*LAMBDA)
                    bqm.add_linear('x_{}_{}'.format(i, p), -LAMBDA)
                bqm.offset += LAMBDA



    return bqm





def solve_tsp(matrix,clusters):
    sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': TOPOLOGY}))
    bqm = create_bqm_GTSP(matrix,clusters)

    sampleset = sampler.sample(bqm, num_reads=NUM_READS, annealing_time=ANNEALING_TIME, label='GTSP')
    #print average energy of samples
    print("Average energy: ", np.mean(sampleset.record.energy))

    embedding = sampleset.info['embedding_context']['embedding']

    print(f"Number of logical variables: {len(embedding.keys())}")
    print(f"Number of physical qubits used in embedding: {sum(len(chain) for chain in embedding.values())}")

    #inspector.show(sampleset)
    return sampleset


def verify_solution_gtsp(solution, matrix, clusters):
    solution = [k for k, v in solution.items() if v == 1]
    solution = [(int(x.split('_')[1]),int(x.split('_')[2])) for x in solution]
    solution_sorted = list(sorted(solution, key=lambda x: x[1]))
    solution_sorted = [x[0] for x in solution_sorted]

    for c in clusters.values():
        count = 0
        for i in c:
            if i in solution_sorted:
                count += 1
        if count != 1:
            return False
    if len(solution) != len(clusters.keys()):
        return False
    return True


def print_sampleset(matrix,sampleset,best,coords,clusters):

    verified_samples = []
    for i, sample in enumerate(sampleset):
        if verify_solution_gtsp(sample, matrix, clusters):
            solution = pretty_print(sample,matrix)
            verified_samples.append(solution)

    print("Number of solutions found: ", len(verified_samples))
    costs = [cost_function(x,matrix) for x in verified_samples]
    if len(verified_samples) == 0:
        print("No solution found")
        return

    print("Best solution: ", verified_samples[np.argmin(costs)])
    print("Cost: ", np.min(costs))
    plot_solution( verified_samples[np.argmin(costs)],matrix,coords)


def create_random_reinsertion_neighbour(path, SIZE,clusters):
    i = random.randint(0, len(path) - 1)

    path_copy = path.copy()
    for c in clusters.values():
        if path[i] in c:
            j = random.choice(c)
            if len(c) == 1:
                return path_copy
            while j == path[i]:
                j = random.choice(c)

    path_copy[i] = j
    return path_copy



def create_random_neighbour(path, SIZE,clusters):
    if random.random() < 0.5:
        return create_random_swap_neighbour(path, SIZE)
    else:
        return create_random_reinsertion_neighbour(path, SIZE,clusters)


def create_random_swap_neighbour(path, SIZE):
    i = random.randint(0, len(path) - 1)
    j = random.randint(0, len(path) - 1)
    path[i], path[j] = path[j], path[i]
    return path


def simulated_annealing_gtsp(matrix, initial_temperature,min_temperature, cooling_rate, max_iterations,clusters,coords):
    s = [cluster[0] for cluster in clusters.values()]
    SIZE = len(matrix)

    best_s = s
    temperature = initial_temperature
    while temperature > min_temperature:
        for i in range(max_iterations):
            s_prime = create_random_neighbour(s, SIZE,clusters)
            delta = cost_function(s_prime,matrix) - cost_function(s,matrix)
            if delta < 0:
                s = s_prime
                if cost_function(s,matrix) < cost_function(best_s,matrix):
                    best_s = s
            else:
                if random.random() < np.exp(-delta/temperature):
                    s = s_prime
        temperature = temperature * cooling_rate

    print("Best solution: ", best_s)
    print("Cost: ", cost_function(best_s,matrix))
    plot_solution(best_s,matrix,coords)
    return best_s


def main():

    matrix,clusters,coords = load_gtsp('data/gtsp_'+str(NUM_CITY)+'.gtsp')
    start = time.time()
    print("Simulated Annealing")
    best = simulated_annealing_gtsp(matrix,100, 0.01, 0.9, 1000, clusters,coords)
    print("Time: ", time.time()-start)
    print("-----------------------------")
    print("Quantum Annealing")
    start = time.time()
    sampleset = solve_tsp(matrix,clusters)
    print_sampleset(matrix,sampleset,best,coords,clusters)
    print("Time: ", time.time()-start)



main()
