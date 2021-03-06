from typing import Tuple

import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Point, box
from sklearn.cluster import KMeans

from statistics.plots import plot_clusters


def get_graph(name, show=True):

    try:
        graph = ox.load_graphml(name)
    except FileNotFoundError:
        graph = download_graph(name)
    if show:
        ox.plot_graph(graph)
    graph = nx.relabel.convert_node_labels_to_integers(graph)
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


def get_map_data(place: str, crs:int) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

    try:
        nodes = gpd.read_file(place+'_nodes.gpkg')
        edges = gpd.read_file(place+'_edges.gpkg')
        return nodes, edges

    except:
        G = get_graph(place)
        if not G:
            exit(-1)
        nodes, edges = ox.graph_to_gdfs(G)
        nodes.to_crs(epsg=crs)
        edges.to_crs(epsg=crs)
        edges = edges[['u', 'v', 'length', 'geometry']]
        nodes = nodes[['osmid', 'y', 'x',  'geometry']]

        return nodes, edges


def cluster_points(points: gpd.GeoDataFrame, num_clusters, geoname, crs, metric_crs,
                   plot_filename=None) -> gpd.GeoDataFrame:

    coords = np.array([points.geometry.x, points.geometry.y]).T
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)
    centroids = kmeans.cluster_centers_

    points['centroid_label'] = kmeans.labels_
    centroids_df = pd.DataFrame(np.array([range(len(centroids)), centroids.T[0], centroids.T[1]]).T,
        columns=['label', 'x', 'y'])
    centroids_df['label'] = centroids_df.label.apply(int)

    centroids_gdf = gpd.GeoDataFrame(centroids_df, geometry=gpd.points_from_xy(centroids_df.x, centroids_df.y))
    centroids_gdf = centroids_gdf.rename(columns={'geometry':geoname}).set_geometry(geoname, crs=crs)
    centroids_proj = centroids_gdf.to_crs(epsg=metric_crs)
    center_proj = find_centroid(points, metric_crs)
    centroids_gdf['from_center'] = centroids_proj[geoname].apply(lambda point: point.distance(center_proj))
    centroids_gdf['node_count'] = centroids_gdf.label.apply(lambda label: len([l for l in kmeans.labels_ if l == label ]))
    # Plot clusters and centroids
    if plot_filename:
        plot_clusters(coords, centroids, kmeans.labels_, plot_filename)

    return centroids_gdf


def find_centroid(points: gpd.GeoDataFrame, metric_crs) -> Point:
    nodes_proj = points.to_crs(epsg=metric_crs)
    bbox = box(*nodes_proj.unary_union.bounds)
    centroid = bbox.centroid

    return centroid


if __name__ == '__main__':

    graph = get_graph('Copenhagen', True, )
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    xs = []
