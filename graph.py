class Graph:
    def __init__(self, vertices, edges, weights):
        self.vertices = vertices # vertices is a set
        self.edges = edges # edges is a set of pairs
        self.weights = weights # weights is a dictionary where the keys are the edges

    def is_hemiltonian_cycle(self, path):
        # path is a list of edges
        cities_from = [edge[0] for edge in path]
        cities_to = [edge[1] for edge in path]
        if sorted(cities_from) != sorted(cities_to) or sorted(cities_from) != sorted(list(self.vertices)):
            return False

        temp = [cities_to[-1]]
        temp.extend(cities_to[:-1])

        if cities_from == temp:
            return True
        else:
            return False