import geopandas as gpd
import numpy as np
import osmnx as ox
import networkx as nx
import pandas as pd
from shapely.geometry import Point, box
from sklearn.cluster import KMeans
from typing import Tuple
from os import path

from plots import plot_clusters


def get_graph(name:str, show=True):

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


def load_nodes_gdf(place: str, geoname:str, crs:int, input_dir: str) -> gpd.GeoDataFrame:

    try:
        nodes = pd.read_csv(path.join(input_dir, place+'_nodes.csv'))
        nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs=crs)
        nodes = nodes.rename(columns={'geometry': geoname}).set_geometry(geoname, crs=crs)
        return nodes           #, edges

    except FileNotFoundError:
        G = get_graph(place)
        if not G:
            exit(-1)
        nodes = ox.graph_to_gdfs(G, edges=False)
        nodes = nodes.rename(columns={'geometry': geoname}).set_geometry(geoname, crs=crs)
        nodes = nodes[['osmid', 'x', 'y', geoname]]
        nodes[['osmid', 'x', 'y']].to_csv(path.join(input_dir, place+'_nodes.csv'), index=False)
        return nodes


def cluster_points(points: gpd.GeoDataFrame, num_clusters: int, geoname: str,
                   crs:int, metric_crs:int, plot_filename=None) -> pd.DataFrame:

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

    centroids_proj['from_center'] = centroids_proj[geoname].apply(lambda point: point.distance(center_proj))
    centroids_proj['node_count'] = centroids_proj.label.apply(lambda label: len([l for l in kmeans.labels_ if l == label]))
    cenroids_proj = centroids_proj.sort_values(by='node_count').reset_index(drop=True)
    centroids_proj = centroids_proj.sort_values(by='from_center').reset_index()
    centroids_proj = centroids_proj.rename(columns={'index': 'weight1'})
    centroids_proj = centroids_proj.sort_values(by='from_center', ascending=False).reset_index()
    centroids_proj = centroids_proj.rename(columns={'index': 'weight2'})
    centroids_proj = centroids_proj.sort_values(by='label').reset_index()
    centroids_proj = centroids_proj.rename(columns={'index': 'weight3'})
    print(centroids_proj.head())
    if plot_filename:
        plot_clusters(coords, centroids, kmeans.labels_, plot_filename)
    return centroids_proj


def find_centroid(points: gpd.GeoDataFrame, metric_crs) -> Point:
    nodes_proj = points.to_crs(epsg=metric_crs)
    bbox = box(*nodes_proj.unary_union.bounds)
    centroid = bbox.centroid

    return centroid