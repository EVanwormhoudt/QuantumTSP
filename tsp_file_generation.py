import numpy as np

def generate_tsp_file(n,max_dist=100, min_dist=10):
    """Generate a TSP file with n cities.

    The cities are placed randomly in a square of side 1000.
    The file is written to filename.

    """

    # Generate random cities with symmetric distance matrix

    filename = 'data/tsp_{}.tsp'.format(n)

    max_len = len(str(max_dist))+1

    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                dist[i, j] = 0
            else:
                dist[i, j] = dist[j, i] = np.random.randint(min_dist, max_dist)




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


generate_tsp_file(5)
