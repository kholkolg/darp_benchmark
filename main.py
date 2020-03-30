import numpy as np
import pandas as ps
import geopandas as gpd
import osmnx as ox
import networkx as nx
from matplotlib.collections import LineCollection
from shapely.geometry import Point, Polygon, LineString
from os import path
import matplotlib.pyplot as plt



def draw_graph(G, nodes=True, edges=True, show=False, path=None):

    if not nodes and not edges:
        nodes = True
    print(G.number_of_nodes(), G.number_of_edges())

    fig, ax = plt.subplots()
    if nodes:
        points = [[d['x'], d['y']] for _, d in G.nodes(data=True)]
        ax.scatter([x for x, _ in points], [y for _, y in points], s=5, marker='o')

    if edges:
        lines = [[[G.nodes[u]['x'], G.nodes[u]['y']], [G.nodes[v]['x'], G.nodes[v]['y']]] for u, v, _ in G.edges]
        lc = LineCollection(lines, linewidths=0.5, colors='black')
        ax.add_collection(lc)

    ax.autoscale()
    if path:
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()


def get_graph(name, show=True):

    try:
        graph = ox.load_graphml(name)
    except FileNotFoundError:
        graph = download_graph(name)
    if show:
        ox.plot_graph(graph)
    return graph


def download_graph(place):
    graph = None
    try:
        graph = ox.graph_from_place(place, network_type='drive', which_result=1, simplify=True)
    except TypeError:
        try:
            graph = ox.graph_from_place(place, network_type='drive', which_result=2, simplify=True)

        except:
            print('Failed to download graph')
            exit(-1)
    ox.save_graphml(graph, place+'.graphml')
    return graph


if __name__ == '__main__':

    graph = get_graph('Copenhagen', True, 'cpg.graphml')
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    xs = []
