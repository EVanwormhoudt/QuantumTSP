from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

def generate_tsp_file(n,max_dist=100, min_dist=10):
    """Generate a TSP file with n cities.

    The cities are placed randomly in a square of side 1000.
    The file is written to filename.

    """

    # Generate random cities with symmetric distance matrix

    filename = 'data/tsp_{}.tsp'.format(n)

    max_len = len(str(max_dist))+1

    cities = np.random.randint(0, max_dist, (n, 2))

    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            dist[i, j] = int(np.sqrt(np.sum((cities[i] - cities[j])**2)))




    with open(filename, 'w') as f:

        f.write(('NAME: {}\n'+
                'TYPE: TSP\n'+
                'COMMENT:{} \n'+
                'DIMENSION: {}\n'+
                'EDGE_WEIGH_TYPE: "EXPLICIT"\n'+
                'EDGE_WEIGHT_FORMAT: "FULL_MATRIX"\n').format(filename, "by Elliott vanwormhoudt", n))

        f.write('\n\nEDGE_WEIGHT_SECTION\n')
        for i in range(n):
            for j in range(n):
                f.write((' {0:'+str(max_len)+'d} ').format(dist[i, j]))
            f.write('\n')



        f.close()



def generate_gtsplib_file(n,max_dist=100, min_dist=10,max_cities_per_cluster=4):


    """Generate a GTSP file with n cities.

    """

    cities_remaining = n
    clusters = []
    clusters_coords = []
    cities = []
    while cities_remaining > 0:
        cluster_size = min(np.random.randint(2,max_cities_per_cluster),cities_remaining)
        clusters.append(cluster_size)
        cities_remaining -= cluster_size



    for i in clusters:
        #generate a random point
        while True:
            coords = np.random.randint(0, max_dist, 2)
            for j in clusters_coords:
                if sqrt((coords[0] - j[0])**2 + (coords[1] - j[1])**2) < 2*min_dist:
                    break
            else:
                break

        clusters_coords.append(coords)
        for j in range(i):
            x = coords[0] + np.random.randint(-min_dist/2,min_dist/2)
            y = coords[1] + np.random.randint(-min_dist/2,min_dist/2)
            if [x,y] in cities:
                continue
            cities.append([x,y])

    filename = 'data/gtsp_{}.gtsp'.format(n)

    plt.plot([i[0] for i in cities],[i[1] for i in cities],'o')
    plt.show()

    max_len = len(str(max_dist))+1



    with open(filename, 'w') as f:

            f.write(('NAME: {}\n'+
                    'TYPE: GTSP\n'+
                    'COMMENT:{} \n'+
                    'DIMENSION: {}\n'+
                    'EDGE_WEIGH_TYPE: "EXPLICIT"\n'+
                    'EDGE_WEIGHT_FORMAT: "COORDS"\n').format(filename, "by Elliott vanwormhoudt", n))


            f.write('\n\nNODE_COORD_SECTION\n')
            index = 0
            for i in range(len(clusters)):
                for j in range(clusters[i]):
                    f.write((' {0:'+str(max_len)+'d} {1:'+str(max_len)+'d} {2:'+str(max_len)+'d} {3:' +str(max_len)+'d}\n').format(index,i+1,cities[index][0],cities[index][1]))
                    index += 1







