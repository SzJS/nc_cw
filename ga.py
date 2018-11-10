import random
import numpy as np

# tours are represented as permutations of the edges
# vertices are a set of strings (labels)
# weights is a matrix containing the weight of each edge

def generate_tour(vertices):
    result = list(vertices)
    random.shuffle(result)
    return result

def generate_initial_tours(vertices, N):
    return [generate_tour(vertices) for i in range(N)]

def get_edges(tour):
    return list(zip(tour, tour[1:] + [tour[1]]))

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

def create_edge_map(tour_a, tour_b, edges):
    """needed for the edge recombination crossover operation"""
    edge_map = dict()
    for (a, b) in edges:
        if not a in edge_map.keys():
            edge_map[a] = []
        edge_map[a].append(b)
    return edge_map

def remove_city(edge_map, city):
    """removes city from the edge map; raises error when edge_map doesn't have city as a key"""
    edge_map.pop(city)
    for city_ in edge_map.keys():
        cities = edge_map[city_]
        if city in cities:
            cities.remove(city)
            edge_map[city_] = cities
    return edge_map

def get_city(edge_map, edges, prev_city, vertices):
    """gets a city from the edge map"""
    # TODO: check if this does what it's supposed to

    poss_cities = edge_map.keys()
    pref_cities = [poss_city for poss_city in poss_cities if (prev_city, poss_city) in edges]
    active_edges = [len(edge_map[pref_city]) for pref_city in pref_cities if len(edge_map[pref_city]) != 0]
    cities = [pref_city for pref_city in pref_cities if len(edge_map[pref_city]) != 0]
    if active_edges == []:
        active_edges = [len(edge_map[poss_city]) for poss_city in poss_cities if len(edge_map[poss_city]) != 0]
        cities = [poss_city for poss_city in poss_cities if len(edge_map[poss_city]) != 0]
        if active_edges == []:
            return random.choice(poss_cities)
    min_edges = min(active_edges)
    min_cities = [city for city in cities if len(edge_map[city]) == min_edges]
    return random.choice(min_cities)

def get_offspring(tour_a, tour_b, vertices):
    """uses ER crossover to create offspring"""

    edges = get_edges(tour_a) + get_edges(tour_b)
    edges = edges + [(v, u) for (u, v) in edges] # edges are undirected
    edges = list(set(edges)) # removing duplicates
    edge_map = create_edge_map(tour_a, tour_b, edges)

    # now we use the edge map to create the tour
    tour = []
    city = random.choice(vertices)
    tour.append(first_city)
    edge_map = remove_city(edge_map, city)
    while len(tour) < len(vertices):
        city = get_city(edge_map, edges, tour[-1], vertices)
        tour.append(city)
        if len(tour) < len(vertices) - 1: # edge_map is empty in the last iteration
            edge_map = remove_city(edge_map, city)
    return tour

def crossover(tour_a, tour_b, cross_prob, vertices):
    if random.random() <= cross_prob:
        offspring_a = get_offspring(tour_a, tour_b, vertices)
        offspring_b = get_offspring(tour_a, tour_b, vertices)
        return (offspring_a, offspring_b)
    else:
        return (tour_a, tour_b) # crossover doesn't happen

def selection(tours, vertices, weights, N):
    """the plain variant of the roulette wheel sampling, implemented using np.random.choice"""
    tour_lengths = [tour_length(tour) for tour in tours]
    max_length = max(tour_lengths)
    fitnesses = [max_length - tour_length for tour_length in tour_lengths]
    total_fitness = sum(fitnesses)
    probs = [fitness / total_fitness for fitness in fitnesses]
    return list(np.random.choice(tours, size=N, replace=True, p=probs))

def pair_up(population):

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
    # TODO: print out minimal length?
    tours = generate_initial_tours(vertices, N)
    generations = 0
    while True:
        generations += 1
        print("generation: ", generations)
        tours = selection(tours, vertices, weights, N) # intermediate pop

        pairs = pair_up(tours)
        pairs = [crossover(pair[0], pair[1], cross_prob) for pair in pairs] # crossover

        tours = [tour for pair in pairs for tour in pair]
        tours = [mutate(tour, pm) for tour in tours] # mutation

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