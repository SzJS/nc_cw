import random
import numpy as np

# tours are represented as permutations of the edges
# vertices are a set of labels (integers)
# weights is a matrix containing the weight of each edge
# note: it's assumed that there is an edge between each vertex

def generate_tour(vertices):
    result = list(vertices)
    random.shuffle(result)
    return result

def generate_initial_tours(vertices, N):
    return [generate_tour(vertices) for i in range(N)]

def get_edges(tour):
    return list(zip(tour, tour[1:] + [tour[0]]))

def tour_length(tour, weights):
    edges = get_edges(tour)
    return sum(weights[edge[0], edge[1]] for edge in edges)

def mutate(tour, mut_prob):
    """a simple swap mutation"""
    # note: the mutation probability is applied only once
    if random.random() <= mut_prob:
        idx = range(len(tour))
        i, j = random.sample(idx, 2) # note: i = j is possible
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def create_edge_map(tour_a, tour_b, vertices):
    """needed for the edge recombination crossover operation;
       returns a (symmetric) matrix of how many times each edge occurs"""

    # we create an edge map (so a matrix for each edge possibe)
    # each value is initialized to be zero
    edge_map = np.array([[0 for vertex in vertices] for vertex in vertices])

    # we then get all edges for both tours
    edges_a = get_edges(tour_a)
    edges_a = edges_a + [(v, u) for (u, v) in edges_a]
    # edges are undirected, so we need the "flipped" edges too

    edges_b = get_edges(tour_b)
    edges_b = edges_b + [(v, u) for (u, v) in edges_b]
    # edges are undirected, so we need the "flipped" edges too

    edges = edges_a + edges_b

    # and finally, we add up all the edges occuring in either tours
    for (u, v) in edges:
        edge_map[u][v] += 1

    return edge_map

def remove_city_from(edge_map, city):
    """removes all edges that leave from the given city (so sets counts to zero)"""

    edge_map[city, :] = 0

    return edge_map

def remove_city_to(edge_map, city):
    """removes all edges that lead the given city (so sets counts to zero)"""

    edge_map[:, city] = 0

    return edge_map

def get_city(edge_map, prev_city, tour, vertices):
    """gets a city from the edge map"""

    edges_from_prev = edge_map[prev_city]
    non_zero_edges = edges_from_prev[edges_from_prev.nonzero()]
    if len(non_zero_edges) != 0:
        # we know that there is an active edge possible
        # because not every count in the edge map was 0

        smallest = np.amin(non_zero_edges)
        possible_next = np.where(edge_map == smallest) # the cities with the smallest number of active edges
        possible_next = zip(possible_next[0], possible_next[1]) # just datatype manipulation
        possible_next = [v for (u, v) in possible_next if u == prev_city] # edges only from the previous city
    else:
        # we have to randomly choose a new city
        possible_next = [city for city in vertices if not city in tour]

    return random.choice(possible_next)

def get_offspring(tour_a, tour_b, vertices):
    """uses ER crossover to create an offspring of two parent tours"""

    edge_map = create_edge_map(tour_a, tour_b, vertices)

    # now we use the edge map to create the tour
    tour = []
    city = random.choice(list(vertices))
    # we remove edges leading to this city (any edge like that would create a loop)
    edge_map = remove_city_to(edge_map, city)
    tour.append(city)
    while len(tour) < len(vertices):
        prev_city = city

        # we get a new city using our edge map
        city = get_city(edge_map, prev_city, tour, vertices)
        tour.append(city)

        # we are no longer interested in edges from previous city or edges to the current city
        # or the edge from the current city to the previous city (as they would create a loop if included)
        edge_map = remove_city_from(edge_map, prev_city)
        edge_map = remove_city_to(edge_map, city)
        edge_map[city, prev_city] = 0

        if len(tour) == len(vertices) - 1:
            # in the last iteration, the edge_map will be equal to zero everywhere,
            # so we cannot use the edge map;
            # however, there is only one possible vertex left so we just add that to the tour;
            # we find this city by looking at which vertex is not yet in the tour (there is just one)
            city = [vertex for vertex in vertices if not vertex in tour][0]
            tour.append(city)

    return tour

def crossover(tour_a, tour_b, cross_prob, vertices):
    if random.random() <= cross_prob:
        offspring_a = get_offspring(tour_a, tour_b, vertices)
        offspring_b = get_offspring(tour_a, tour_b, vertices)
        return (offspring_a, offspring_b)
    else:
        return (tour_a, tour_b) # crossover doesn't happen in this case

def selection(tours, tour_lengths, vertices, weights, N):
    """the plain variant of the roulette wheel sampling, implemented using np.random.choice"""

    # fitnesses are relative to the worst tour
    max_length = max(tour_lengths)
    fitnesses = [max_length - tour_length for tour_length in tour_lengths]

    # probabilities are just the fitness divided by the total fitness
    total_fitness = sum(fitnesses)
    probs = [fitness / total_fitness for fitness in fitnesses]

    # we do the sampling below using np.random.choice
    # we need to use placeholders, because np.random.choice cannot take arrays of arrays
    tours_placeholder = list(range(len(tours)))
    indices = list(np.random.choice(tours_placeholder, size=N, replace=True, p=probs))
    return [tours[index] for index in indices]

def pair_up(population):
    """helper function for pairing up different tours to undergo crossover"""

    def pop_random(lst):
        idx = random.randrange(0, len(lst))
        return lst.pop(idx)

    pairs = []
    while population:
        rand1 = pop_random(population)
        rand2 = pop_random(population)
        pair = rand1, rand2
        pairs.append(pair)
    return pairs

def genetic_alg(N, cross_prob, mut_prob, vertices, weights, max_generations=1000):
    tours = generate_initial_tours(vertices, N)
    generations = 0
    while True:
        generations += 1
        print("generation: ", generations)
        tour_lengths = [tour_length(tour, weights) for tour in tours]
        print("minimum tour length: ", min(tour_lengths))
        tours = selection(tours, tour_lengths, vertices, weights, N) # intermediate pop

        pairs = pair_up(tours)
        pairs = [crossover(pair[0], pair[1], cross_prob, vertices) for pair in pairs] # crossover

        tours = [tour for pair in pairs for tour in pair]
        tours = [mutate(tour, mut_prob) for tour in tours] # mutation

        if generations >= max_generations: # two: a threshold is reached for the number of iterations
            print("maximum number of generations reached")
            return tours

def woac(tours, vertices):
    """returns an aggregated tour based on the tours given"""
    occurance = np.zeros((len(vertices), len(vertices)))
    # we initialize the occurance matrix to be zero (it's |V| x |V|)
    for tour in tours:
        edges = get_edges(tour)
        for edge in edges:
            occurance[edge[0], edge[1]] += 1
            occurance[edge[1], edge[0]] += 1 # we increment both ways, because matrix is symmetric

    # we start with the most occuring edge
    # to find this edge, we find the index of the maximal value in the occurance matrix
    u, v = np.unravel_index(np.argmax(occurance, axis=None), occurance.shape)

    # we avoid reusing this edge by setting it -1
    occurance[u, v], occurance[v, u] = -1, -1

    result = [] # edges that have been used twice will be put here

    # we then find the most occuring edge adjacent to one of the vertices of the previous edge
    value_u, value_v = np.amax(occurance[u]), np.amax(occurance[v])
    if value_u >= value_v:
        next_vertex = np.argmax(occurance[u], axis=None)
        result.append(u)
        occurance[u, :], occurance[:, u] = -1, -1 # we avoid visiting u again
        last_vertex = v # we know that v has to be the last city in our tour
        # because we already know that we will use v as the last vertex, we don't want to visit it again
        occurance[v, :], occurance[:, v] = -1, -1
    else:
        next_vertex = np.argmax(occurance[v], axis=None)
        result.append(v)
        occurance[v, :], occurance[:, v] = -1, -1 # we avoid visiting v again
        last_vertex = u # we know that u has to be the last city in our tour
        # because we already know that we will use u as the last vertex, we don't want to visit it again
        occurance[u, :], occurance[:, u] = -1, -1
    while len(result) < len(vertices) - 1:
        prev_vertex = next_vertex
        result.append(prev_vertex)
        next_vertex = np.argmax(occurance[prev_vertex], axis=None)
        occurance[prev_vertex, :], occurance[:, prev_vertex] = -1, -1

    result.append(last_vertex)

    return result

def local_search(tour, weights, max_iterations):
    for i in range(max_iterations):
        old_tour = tour[:]
        idx = range(len(tour))
        i, j, k = random.sample(idx, 3)
        tour[i], tour[j], tour[k] = tour[j], tour[k], tour[i]
        old_length = tour_length(old_tour, weights)
        new_length = tour_length(tour, weights)
        if new_length > old_length:
            tour = old_tour # we only revert to the old tour, if the new one is worse
    return tour

def local_search_post_processing(tours, weights, max_iterations):
    tours = [local_search(tour, weights, max_iterations) for tour in tours]
    lengths = [tour_length(tour, weights) for tour in tours]
    idx = lengths.index(min(lengths))
    return tours[idx]

def get_weight_matrix(n, lower_bound=1, upper_bound=100):
    weights = np.random.randint(lower_bound, upper_bound+1, (n, n))
    weights = (weights + weights.T) # we make the matrix symmetric
    np.fill_diagonal(weights, 0) # any vertex has a distance of 0 to itself
    return weights