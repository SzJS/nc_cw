import random

def generate_genes(length):
    return "".join([str(random.randint(0, 1)) for i in range(length)])

def generate_initial_pop(length, N):
    return [generate_genes(length) for i in range(N)]

# def max_fitness(length):
#     return (2 ** length) - 1

# def fitness(genes):
#     return int(genes, 2)

def mutate(genes, pm):
    genes = list(genes) # making the string mutable
    for i in range(len(genes)):
        gene = genes[i]
        if random.random() <= pm:
            if gene == "0":
                genes[i] = "1"
            elif gene == "1":
                genes[i] = "0"
    return "".join(genes) # we return the right datatype (string)

def crossover(genes_a, genes_b, pc):
    if random.random() <= pc:
        point = random.randrange(start=1, stop=len(genes_a))
        genes_a, genes_b = list(genes_a), list(genes_b)
        return ("".join(genes_a[:point]) + "".join(genes_b[point:]),
                "".join(genes_b[:point]) + "".join(genes_a[point:]))
    else:
        return (genes_a, genes_b) # crossover doesn't happen

def selection(population, N, fitness):
    fitnesses = [fitness(genes) for genes in population]
    sequence = [fitnesses[i] * [population[i]] for i in range(N)]
    sequence = [item for subsequence in sequence for item in subsequence] # flattening the list
    return list(random.sample(sequence, N))

def pop_random(lst): # helper function for pair_up
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)

def pair_up(population):
    pairs = []
    while population:
        rand1 = pop_random(population)
        rand2 = pop_random(population)
        pair = rand1, rand2
        pairs.append(pair)
    return pairs

def simulate(N, pc, pm, length, fitness, max_fitness=None, max_generations=None):
    population = generate_initial_pop(length, N)
    generations = 0
    while True:
        generations += 1
        print("generation: ", generations)
        population = selection(population, N, fitness) # intermediate pop

        pairs = pair_up(population)
        pairs = [crossover(pair[0], pair[1], pc) for pair in pairs] # crossover

        population = [genes for pair in pairs for genes in pair]
        population = [mutate(genes, pm) for genes in population] # mutation

        fitnesses = [fitness(genes) for genes in population]
        current_max = max(fitnesses)
        print("maximum fitness is: ", current_max)
        
        # there are two ways for this algorithm to terminate
        if max_fitness: # one: optimal solution is found (if optimal fitness is known)
            if current_max == max_fitness:
                print("maximal fitness found")
                return
        if generations == max_generations: # two: a threshold is reached for the number of iterations
            print("maximum number of generations reached")
            return
