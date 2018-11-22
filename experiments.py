import random
import numpy as np
from ga import *
def run_exp(N, n, cross_prob, mut_prob, integers=False):
    vertices = set(range(n))
    weights = get_weight_matrix(n, integers=integers)
    max_generations = 50
    gas = []; islands = []; woacs = []
    for i in range(10):
        pop1 = genetic_alg(int(N / 2), cross_prob, mut_prob, vertices, weights, max_generations)
        pop2 = genetic_alg(int(N / 2), cross_prob, mut_prob, vertices, weights, max_generations)
        pop = pop1 + pop2
        pop = genetic_alg(N, cross_prob, mut_prob, vertices, weights, 1, tours=pop)
        tour_lengths = [tour_length(tour, weights) for tour in pop]
        island_length = min(tour_lengths)
        pop = genetic_alg(N, cross_prob, mut_prob, vertices, weights, max_generations)
        tour_lengths = [tour_length(tour, weights) for tour in pop]
        min_length = min(tour_lengths)
        mean_length = sum(tour_lengths) / len(tour_lengths)
        max_length = max(tour_lengths)
        woac_tour = woac(pop, vertices)
        woac_length = tour_length(woac_tour, weights)
        gas.append(min_length); islands.append(island_length); woacs.append(woac_length)
        print("---{:.1f}---".format(i+1))
        print("ga: {:.1f} | {:.1f} | {:.1f}".format(min_length, mean_length, max_length))
        print("island: {:.1f}".format(island_length))
        print("woac: {:.1f}".format(woac_length))
    ga_mean = sum(gas) / 10
    islands_mean = sum(islands) / 10
    woac_mean = sum(woacs) / 10
    print("=== MEAN ===")
    print("ga (min): {:.1f}".format(ga_mean))
    print("island: {:.1f}".format(islands_mean))
    print("woac: {:.1f}".format(woac_mean))

# Bad condition

run_exp(n=11, N=20, cross_prob=0.7, mut_prob=0.01)

run_exp(n=11, N=20, cross_prob=0.7, mut_prob=0.05)

run_exp(n=11, N=200, cross_prob=0.7, mut_prob=0.01)

# Good condition

run_exp(n=222, N=20, cross_prob=0.7, mut_prob=0.01)

run_exp(n=222, N=20, cross_prob=0.7, mut_prob=0.05)

run_exp(n=222, N=200, cross_prob=0.7, mut_prob=0.01)


# Intermediate condition

run_exp(n=44, N=20, cross_prob=0.7, mut_prob=0.01)

run_exp(n=44, N=20, cross_prob=0.7, mut_prob=0.05)

run_exp(n=44, N=200, cross_prob=0.7, mut_prob=0.01)