import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns



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
    s = np.random.randint(0, len(matrix), len(matrix))
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