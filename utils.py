import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from    sklearn import manifold




def cost_function(x, matrix):
    return np.sum(np.fromiter((matrix[x[i], x[i + 1]] for i in range(len(x) - 1)), dtype=float)) + matrix[x[-1], x[0]]


def create_random_neighbour(path, SIZE):
    i = random.randint(0, SIZE - 1)
    j = random.randint(0, SIZE - 1)
    while j == i:
        j = random.randint(0, SIZE - 1)

    path_copy = deepcopy(path)
    tmp = path_copy[i]
    path_copy[i] = path_copy[j]
    path_copy[j] = tmp
    return path_copy


def simulated_annealing(matrix, initial_temperature,min_temperature, cooling_rate, max_iterations):
    s = list(range(len(matrix)))
    SIZE = len(s)
    best_s = s
    temperature = initial_temperature
    while temperature > min_temperature:
        for i in range(max_iterations):
            s_prime = create_random_neighbour(s, SIZE)
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
    return best_s


def load_tsp(filename: str):
    """Load a TSP file and return the distance matrix.

    The file must be in TSPLIB format.

    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    start,n = 0,0
    for i, line in enumerate(lines):
        if line.startswith('DIMENSION'):
            n = int(line.split(':')[-1])
        if line.startswith('EDGE_WEIGHT_SECTION'):
            start = i + 1

    end = start + n

    dist = np.zeros((n, n), dtype=int)
    for i, line in enumerate(lines[start:end]):
        dist[i, :] = np.array([int(x) for x in line.split()])

    return dist

def pretty_print(solution, matrix):

    solution = [k for k, v in solution.items() if v == 1]
    solution = [(int(x.split('_')[1]),int(x.split('_')[2])) for x in solution]
    solution_sorted = list(sorted(solution, key=lambda x: x[1]))
    solution_sorted = [x[0] for x in solution_sorted]
    print(solution_sorted)
    print(cost_function(solution_sorted, matrix))
    return solution_sorted


def verify_solution(solution, matrix):
    solution = [k for k, v in solution.items() if v == 1]
    solution = [(int(x.split('_')[1]),int(x.split('_')[2])) for x in solution]
    if len(solution) != len(matrix):
        return False
    if list(range(len(matrix))) != list(sorted([x[0] for x in solution])):
        return False
    return True

def plot_solution(solution, matrix,best=False):

    mds_model = manifold.MDS(n_components=2, random_state=17,
                             dissimilarity='precomputed',normalized_stress=False)
    mds_fit = mds_model.fit(matrix)
    mds_coords = mds_model.fit_transform(matrix)
    color = 'b'
    if best:
        color = 'g'

    plt.figure(figsize=(10, 10))
    plt.title("Visualisation of the solution")
    # add the cost
    plt.suptitle("Cost: {}".format(cost_function(solution, matrix)))
    plt.scatter(mds_coords[:, 0], mds_coords[:, 1], s=100, c='r')
    plt.plot(mds_coords[solution, 0], mds_coords[solution, 1], c=color)
    plt.plot([mds_coords[solution[-1], 0], mds_coords[solution[0], 0]], [mds_coords[solution[-1], 1], mds_coords[solution[0], 1]], c=color)
    plt.show()




