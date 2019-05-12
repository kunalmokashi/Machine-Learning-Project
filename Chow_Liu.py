from collections import defaultdict, OrderedDict, Counter
import numpy as np
import time

class Graph():
    def __init__(self, no_vertices):
        self.graph = defaultdict(list)
        self.vertex = no_vertices
        self.weight = defaultdict()
        self.parent = [-1] * no_vertices
        self.rank = [0]*no_vertices
        self.visited = np.zeros(no_vertices, dtype=bool)

    def __getneighbors__(self, v):
        neighbors = []
        for neighbor in self.graph[v]:
            if self.visited[neighbor] == False:
                neighbors.append(neighbor)
        return neighbors

    def add_edge(self, n_u, n_v, e_weight=None, add_reverse=False):
        self.graph[n_u].append(n_v)
        self.weight[(n_u, n_v)] = e_weight
        if add_reverse:
            self.graph[n_v].append(n_u)
            self.weight[(n_v, n_u)] = e_weight

    def find(self, node):
        if self.parent[node] == -1:
            return node
        return self.find(self.parent[node])

    def union(self, n_x, n_y):
        x = self.find(n_x)
        y = self.find(n_y)
        if self.rank[x] < self.rank[y]:
            self.parent[x] = y
        elif self.rank[y] < self.rank[x]:
            self.parent[y] = x
        else:
            self.parent[x] = y
            self.rank[y] += 1

    def MST_kruskal(self):
        MST = OrderedDict()
        s, p = 0, 0
        temp = sorted(self.weight.items(), key=lambda v: v[1])
        while p < self.vertex-1:
            n_x, n_y, e_w = temp[s][0][0], temp[s][0][1], self.weight.get((temp[s][0]))
            s += 1
            x = self.find(n_x)
            y = self.find(n_y)
            if x != y:
                p += 1
                MST[(n_x, n_y)] = e_w
                self.union(x, y)
        return MST


# build a complete graph given the feature set.
def build_complete_graph(data):
    start = time.time()
    graph = Graph(len(data.columns))
    print("Build complete graph")
    for index_1, feature_1 in enumerate(list(data.columns.values)):
        for index_2, feature_2 in enumerate(list(data.columns.values)):
            if feature_1 == feature_2:
                #it is the same column, continue to the next feature.
                continue
            else:
                # for each edge in the graph find the mutual information score
                # edge weight = negative score.
                score = find_mutual_information_score(data, feature_1, feature_2)
                graph.add_edge(index_1, index_2, -score)
    end = time.time()
    print("Time taken to build complete graph with n = ", len(data.columns), " is ", (end - start))
    return graph

# build a max spanning tree using chow liu algorithm.
def run_chow_liu(data):
    #first build a complete graph from the data.
    graph = build_complete_graph(data)
    print("Complete graph built with mutual information between edges - ", graph)
    # Find Maximum spanning tree using kruskal's algorithm
    MST = graph.MST_kruskal()
#    print("MST - ", MST)
#    MST_graph = Graph(len(MST))
#    for key, value in MST.items():
#        MST_graph.add_edge(key[0], key[1], value)
    return MST

# find mutual information score between nodes (features)
def find_mutual_information_score(data, feature_a, feature_b):
    number_of_samples = len(data.index)
    p_feature_a = defaultdict(float)
    p_feature_b = defaultdict(float)
    p_feature_a_feature_b = defaultdict(float)

    column_a = data[feature_a]
    column_a = np.nan_to_num(column_a)
    count_a_dict = Counter(column_a)
    for key, value in count_a_dict.items():
        p_feature_a[key] =  value / number_of_samples

    column_b = data[feature_b]
    column_b = np.nan_to_num(column_b)
    count_b_dict = Counter(column_b)
    for key, value in count_b_dict.items():
        p_feature_b[key] =  value / number_of_samples

    column_a_column_b = list(zip(column_a, column_b))
    count_a_b_dict = Counter(column_a_column_b)
    for key, value in count_a_b_dict.items():
        p_feature_a_feature_b[key] =  value / number_of_samples
    score = 0
    for tuples in column_a_column_b:
        try:
            score += np.log(p_feature_a_feature_b[tuples] / (p_feature_a[tuples[0]] * p_feature_b[tuples[1]]))
        except ZeroDivisionError:
            score += 0
    return score / number_of_samples
