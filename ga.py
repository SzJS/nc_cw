import math, random
import numpy as np

# tours are represented as permutations of the edges
# vertices are a set of labels (integers)
# weights is a matrix containing the weight of each edge
# note: it's assumed that there is an edge between each vertex

def generate_tour(vertices):
    """generates a tour from the given vertices, such that first_city is always the first in the tour"""
    result = list(vertices)
    random.shuffle(result)
    first_city = 0 # 0 is always the first city
    result = [city for city in result if city != first_city]
    result = [first_city] + result
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
        idx = range(1, len(tour)) # we never swap the first city (which is 0)
        i, j = random.sample(idx, 2) # note: i = j is possible
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def get_active(city, edges, possible_cities):
    """gets the number of active edges for a given city"""
    return sum(1 for poss_city in possible_cities if (city, poss_city) in edges)

def get_city(prev_city, tour, edges, vertices):
    """gets a city from the edge map"""

    possible_cities = [city for city in vertices if not city in tour]
    next_cities = [next_city for (city, next_city) in edges if city == prev_city and next_city in possible_cities]
    if next_cities == []:
        min_cities = possible_cities # we need to make a random choice
    else:
        edge_map = [get_active(city, edges, possible_cities) for city in next_cities]
        min_act_edges = min(edge_map)
        min_cities = [next_cities[i] for i in range(len(next_cities))
                      if edge_map[i] == min_act_edges]

    return random.choice(min_cities)

def get_offspring(tour_a, tour_b, vertices):
    """uses ER crossover to create an offspring of two parent tours"""
    
    edges = get_edges(tour_a) + get_edges(tour_b)
    edges = edges + [(v, u) for (u, v) in edges] # edges are symmetric
    edges = set(edges) # duplicates are ignored

    first_city = 0
    tour = [first_city]
    city = first_city
    
    while len(tour) < len(vertices):
        prev_city = city
        city = get_city(prev_city, tour, edges, vertices)
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

def genetic_alg(N, cross_prob, mut_prob, vertices, weights, max_generations=1000, should_print=False, tours=None):
    if tours == None:
        tours = generate_initial_tours(vertices, N)
    generations = 0
    while True:
        generations += 1
        tour_lengths = [tour_length(tour, weights) for tour in tours]
        tours = selection(tours, tour_lengths, vertices, weights, N) # intermediate pop

        pairs = pair_up(tours)
        pairs = [crossover(pair[0], pair[1], cross_prob, vertices) for pair in pairs] # crossover

        tours = [tour for pair in pairs for tour in pair]
        tours = [mutate(tour, mut_prob) for tour in tours] # mutation
        
        if should_print:
            print("generation: ", generations)
            print("min tour: ", min(tour_lengths))

        if generations >= max_generations:
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

def get_weight_matrix(n, lower_bound=1, upper_bound=100, integers=True):
    if integers:
        weights = np.random.randint(lower_bound, upper_bound+1, (n, n))
        weights = (weights + weights.T) # we make the matrix symmetric
        np.fill_diagonal(weights, 0) # any vertex has a distance of 0 to itself
        return weights
    else:
        points = np.random.randint(0, upper_bound, (n, 2)) + np.random.rand(n, 2)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = math.hypot(points[i, 0] - points[j, 0], points[i, 1] - points[j, 1])
                distances[j, i] = distances[i, j]
        return distances