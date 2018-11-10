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
            # so we cannot use the edge map, but there is only one possible vertex left
            # so we just add that to the tour;
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

def makes_loop(edges, new_edge):
    edges = edges + new_edge
    firsts = [edge[0] for edge in edges]
    seconds = [edge[1] for edge in edges]

    vertices = list({v for edge in edges for v in edge})
    visited = []

    v = vertices.pop()
    while len(vertices) > 0:
        visited.append(v)
        if v in firsts:
            i = firsts.index(v)
            edge = edges[i]
            v = edge[1]
            vertices.remove(v)
        elif v in seconds:
            i = seconds.index(v)
            edge = edges[i]
            v = edge[0]
            vertices.remove(v)
        else:
            v = vertices.pop() # we still have vertices we haven't visited, but that may not be connected
        if v in visited:
            return True # if we get a vertex we have already seen, we have a loop

    return False


def woac(tours, vertices, weights):
    """returns an aggregated tour based on the tours given"""
    occurences = dict()
    for tour in tours:
        edges = get_edges(tour)
        for edge in edges:
            edge = (min(edge), max(edge)) # we "sort" the edge because (a, b) = (b, a)
            if not edge in occurences.keys():
                occurences[edge] = 1
            else:
                occurences[edge] += 1

    result = []

    while len(result) < len(vertices):
        values = occurences.values()
        values.sort() # we sort in non-decreasing order
        i = 0
        while True: # should eventually find a good value, hence the infinite loop
            ith_most_occuring = values[-i-1]
            edge = edges.keys()[values.index(ith_most_occuring)]
            if not makes_loop(result, edge):
                result.append(edge)
                occurences.pop(edge) # we want to avoid reusing the same edge, so we remove it
                # TODO: once a vertex is added twice, we could maybe remove every edge containing it
                break
            elif: # we know we have a loop, so we just need to check if we have everything twice
                vs = [v for e in (edges + edge) for v in e]
                vs.sort()
                if vs == sorted(list(vertices) + list(vertices)): # every element occurs twice, so we are done
                    result.append(edge)
                    break

    return result