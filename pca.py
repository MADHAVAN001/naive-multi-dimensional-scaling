import numpy as np
import pickle
import os
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt


class Graph:

    def __init__(self, name):
        self.nodes = list()
        self.adjacency_dict = dict()
        self.distance = np.array((1, 1))
        self.name = name
        self.nodes_pkl = "nodes_{}.pkl".format(self.name)
        self.distance_pkl = "distance_{}.pkl".format(self.name)
        self.adjacency_pkl = "adjacency_{}.pkl".format(self.name)

    def build_graph(self, x_list, k):
        if os.path.exists(self.nodes_pkl) and os.path.exists(self.adjacency_pkl):
            self.nodes = pickle.load(open(self.nodes_pkl, "rb"))
            self.adjacency_dict = pickle.load(open(self.adjacency_pkl, "rb"))
            return

        self.nodes = list()
        self.adjacency_dict = dict()

        i = 0
        for point in x_list:
            self.nodes.append((i, point))
            i += 1

        for point in self.nodes:
            self.connect_k_nearest_neighbours(point, k)

        pickle.dump(self.nodes, open(self.nodes_pkl, "wb"))
        pickle.dump(self.adjacency_dict, open(self.adjacency_pkl, "wb"))

    def connect_k_nearest_neighbours(self, point, k):

        distance_tuples = list()
        for node in self.nodes:
            distance_tuples.append((node[0], np.linalg.norm(point[1] - node[1])))

        distance_tuples.sort(key=lambda tup: tup[1])
        distance_tuples = distance_tuples[1:min(len(distance_tuples), k + 1)]
        self.adjacency_dict[point[0]] = distance_tuples

    def all_pairs_shortest_paths(self):
        if os.path.exists(self.distance_pkl):
            self.distance = pickle.load(open(self.distance_pkl, "rb"))
            return

        self.distance = float(np.inf) * np.ones((len(x), len(x)))

        for key, values in self.adjacency_dict.items():
            for edge in values:
                self.distance[key][edge[0]] = edge[1]
            self.distance[key][key] = 0

        for k in range(len(self.nodes)):
            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if self.distance[i][j] > self.distance[i][k] + self.distance[k][j]:
                        self.distance[i][j] = self.distance[i][k] + self.distance[k][j]

        pickle.dump(self.distance, open(self.distance_pkl, "wb"))

    def mds(self, k, epsilon=0.001):

        old_stress = np.inf

        Y = np.random.rand(len(self.nodes) * k)
        Y = Y.reshape((len(self.nodes), k))
        stress = 0
        i = 0
        while abs(old_stress - stress) > epsilon:
            old_stress = stress
            i += 1
            norm = np.zeros((len(self.nodes), len(self.nodes)))
            for i in range(Y.shape[0]):
                for j in range(Y.shape[0]):
                    norm[i][i] = np.linalg.norm(Y[i] - Y[j])

            # Compute stress
            stress = ((norm.ravel() - self.distance.ravel()) ** 2).sum()

            norm[norm == 0] = 1e-5
            ratio = self.distance / norm
            B = - ratio
            B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)
            Y = 1. / len(self.nodes) * np.dot(B, Y)

            print("{}:{}".format(i, stress))

            if i > 300:
                break

        return Y


def get_points(file_name):
    points_list = list()
    with open(file_name) as points:
        line = points.readline()
        while line:
            numbers_list = list(map(float, line.strip().split()))
            points_list.append(np.array(numbers_list, dtype=np.float))
            line = points.readline()
    return points_list


if __name__ == '__main__':
    # x = [np.array([0, 0]), np.array([0, 2]), np.array([2, 2])]
    # graph = Graph()
    # graph.build_graph(x, 3)
    # graph.all_pairs_shortest_paths()
    # graph.mds(1)

    x = get_points("swiss_roll_hole.txt")

    plt_array = np.array(x)

    t = plt_array[:, 0]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(
        plt_array[:, 0],
        plt_array[:, 1],
        plt_array[:,1],
        cmap='viridis',
        linewidth=0.5
    )

    plt.show()

    graph = Graph("swiss_roll_hole")
    graph.build_graph(x, 10)
    graph.all_pairs_shortest_paths()
    y = graph.mds(2)

    plt.scatter(y[:, 0], y[:, 1])
    plt.show()
