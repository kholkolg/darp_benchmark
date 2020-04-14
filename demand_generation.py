import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from sklearn.cluster import KMeans
from typing import List, Tuple, Iterable
from scipy.spatial import cKDTree
import sys

from demand_generation_helpers import (load_nodes_gdf, save_trips_csv,
                                       save_shapefiles, prepare_config)


def generate_trips(case_name=None):


    """
    Generates trips for the given map.
    Map is read from input_dir/map_name.csv,
    if .csv file doesn't exist, map will be downloaded with osmnx.
    Result is trips.csv (time_ms, origin, dest) and map_name_nodes.csv (id, osmid, x, y)
    separated by '\t' for benchmark, and .shp files for qgis.
    :param case_name: case name from config.ini file, if none, the default parameters are used
    :return:
    """
    params = prepare_config(case_name)
    map_name = params['city']
    num_trips = int(params['numRequests'])
    min_dist = float(params['minDistance'])
    avg_dist = float(params['avgDistance'])
    input_dir = params['inputDir']
    output_dir = params['outputDir']
    unused_roads = params['unusedRoads'].split(';')
    cluster_size = float(params['clusterSize'])
    crs_geo = int(params['crsGeo'])
    crs_metric = int(params['crsMetric'])
    peak_hours = [[float(x) for x in p.split(',')]for p in params['peaks'].split(';')]
    save_shp = bool(params['saveShapefiles'])

    all_nodes = load_nodes_gdf(map_name, crs_geo, input_dir)
    # remove nodes located on highways, reindex
    nodes = all_nodes[~all_nodes['highway'].isin(unused_roads)]
    nodes = nodes.reset_index(drop=True)

    nodes = nodes.to_crs(crs=crs_metric)
    n_clusters = num_clusters(nodes, crs_metric, cluster_size)
    centroids = cluster_points(nodes, n_clusters)
    trips = generate_instance(num_trips, nodes, centroids, min_dist, avg_dist, peak_hours)

    # translate back to original indices
    trips['origin'] = nodes.iloc[trips.origin].id.values
    trips['dest'] = nodes.iloc[trips.dest].id.values
    # save generated instance
    save_trips_csv(trips, output_dir)
    # save shapefiles
    if save_shp:
        save_shapefiles(trips, all_nodes, crs_geo, output_dir)

    return all_nodes, trips


def generate_instance(num_requests: int, nodes: gpd.GeoDataFrame, centroids: pd.DataFrame,
                      min_dist: float, avg_dist: float, peak_hours: List[List[float]]):

    # add some more trips to remove too short trips
    num_requests_ = int(1.1*num_requests)
    num_peaks = len(peak_hours)
    num_trips_p = 0 if num_peaks == 0 else int(num_requests_*0.4)
    num_trips_np = num_requests_ - num_trips_p
    columns = ['time_ms', 'cluster1', 'cluster2', 'dist', 'dx', 'dy', 'origin', 'dest']
    all_trips = pd.DataFrame(columns=columns)

    # generante non-peak demand
    # generate request time and trip distance
    all_trips['time_ms'] = generate_time(4, 20, num_trips_np, 95)
    all_trips['dist'] = np.random.normal(avg_dist, avg_dist/2, size=num_trips_np)
    mask = all_trips.dist < min_dist
    all_trips.loc[mask, 'dist'] = np.random.uniform(min_dist, avg_dist, size=sum(mask))
    # origin and destination clusters, direction vector
    all_trips['cluster1'] = select_clusters(centroids.label.values, num_trips_np, np.exp(centroids.node_count/100))
    all_trips['cluster2'] = select_clusters(centroids.label.values, num_trips_np, np.exp(centroids.node_count/100))
    all_trips = cluster_to_vector(all_trips, centroids)
    # select nodes
    all_trips = select_nodes(all_trips, nodes)

    # peak demand
    if num_peaks != 0:
        peak_n = num_trips_p // num_peaks
        probs = [centroids.from_center, 1/centroids.from_center]

        for start, end in peak_hours:
            # higher probabilities for:
            # morning pickup/evening dropoff - further from center
            # evening pickup/morning dropoff - closer to center
            probs_p = probs[0] if start <= 12 else probs[1]
            probs_d = probs[1] if start > 12 else probs[0]

            new_trips = pd.DataFrame(columns=columns)
            # times, distances
            new_trips['time_ms'] = generate_time(start, end, peak_n, 95)
            new_trips['dist'] = np.random.normal(avg_dist, avg_dist / 2, size=peak_n)
            mask = (new_trips.dist < min_dist)
            new_trips.loc[mask, 'dist'] = np.random.uniform(min_dist, avg_dist, size=sum(mask))
            # clusters, directions, nodes
            new_trips['cluster1'] = select_clusters(centroids.label.values, peak_n, probs_p)
            new_trips['cluster2'] = select_clusters(centroids.label.values, peak_n, probs_d)
            new_trips = cluster_to_vector(new_trips, centroids)
            new_trips = select_nodes(new_trips, nodes)
            # add to main dataframe
            all_trips = pd.concat([all_trips, new_trips], axis=0)

    #filter out too short trips
    all_trips['dist'] = all_trips.apply(
        lambda t: nodes.iloc[t.origin].geometry.distance(nodes.iloc[t.dest].geometry),
                                        axis=1)
    all_trips = all_trips[all_trips.dist >= min_dist]
    all_trips = all_trips[(all_trips['time_ms'] >= 0) & (all_trips['time_ms'] < 24*36e5)]
    all_trips = all_trips.reset_index(drop=True)

    # if more than required trips were generated, remove random rows
    num_generated = all_trips.shape[0]
    num_to_remove = num_generated - num_requests
    if num_to_remove > 0:
        to_remove = np.random.choice(all_trips.index, size=num_to_remove, replace=False)
        all_trips = all_trips.drop(to_remove)
    # sort by pickup time and reindex
    all_trips = all_trips.sort_values(by='time_ms').reset_index(drop=True)

    return all_trips


def cluster_to_vector(trips, centroids):
    trips['dx'] = centroids.iloc[trips.cluster2].x.values - centroids.iloc[trips.cluster1].x.values
    trips['dy'] = centroids.iloc[trips.cluster2].y.values - centroids.iloc[trips.cluster1].y.values
    # origin and destination in the same cluster
    mask = (trips.dx == 0) & (trips.dy == 0)
    trips.loc[mask, 'dx'] = np.random.uniform(-1, 1, size=sum(mask))
    trips.loc[mask, 'dy'] = np.random.uniform(-1, 1, size=sum(mask))
    trips['norm'] = np.sqrt(trips.dx ** 2 + trips.dy ** 2)
    trips['dx'] = trips['dx'] / trips['norm']
    trips['dy'] = trips['dy'] / trips['norm']
    trips = trips.drop(columns=['norm'])
    return trips


def select_nodes(trips, nodes):
    """
    Select nodes from nodes dataframe for cluster labels in trips.
    :param trips:
    :param nodes:
    :return:
    """

    p_clusters = trips.cluster1.unique()
    for c in p_clusters:
        cluster_nodes = nodes[nodes.centroid_label == c].index
        mask = trips.cluster1 == c
        trips.loc[mask, 'origin'] = np.random.choice(cluster_nodes, size=sum(mask), replace=True)

    trips['x2'] = nodes.iloc[trips['origin']].geometry.x.values + trips.dx*trips.dist
    trips['y2'] = nodes.iloc[trips['origin']].geometry.y.values + trips.dy*trips.dist

    tree = cKDTree(np.array((nodes.geometry.x, nodes.geometry.y)).T)
    dist, idx = tree.query(np.array((trips.x2, trips.y2)).T, k=1, n_jobs=-1)
    trips['dest'] = idx
    trips = trips.drop(columns=['x2', 'y2'])
    return trips


def select_clusters(points, num_points, probs=None):
    probs = probs if probs is not None else np.ones(len(points))
    if np.sum(probs) != 1:
        probs = probs/np.sum(probs)
    result = np.random.choice(points, size=num_points, p=probs)
    return result


def cluster_points(points: gpd.GeoDataFrame, num_clusters: int) -> pd.DataFrame:
    """
    Cluster points, add 'label column with centroid label to points dataframe.
    Returns dataframe with clusters' centroids (label, x, y, node_count, from_center)
    where node count is the number of nodes in the cluster,
    and 'from_center' is distance of between the cluster centroid and the city center in meters.

    :param points: geodataframe with 'geometry' column
    :param num_clusters: number of required clusters
    :return:
    """
    crs = points.crs
    coords = np.array([points.geometry.x, points.geometry.y]).T
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs=-1).fit(coords)
    centroids = kmeans.cluster_centers_

    points['centroid_label'] = kmeans.labels_
    df = pd.DataFrame(np.array([range(len(centroids)), centroids.T[0], centroids.T[1]]).T,
                                columns=['label', 'x', 'y'])
    df['label'] = df.label.apply(int)

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y),crs=crs)
    city_center = box(*points.unary_union.bounds).centroid
    gdf['from_center'] = gdf.geometry.apply(lambda point: point.distance(city_center))
    gdf['node_count'] = gdf.label.apply(lambda label: len([l for l in kmeans.labels_ if l == label]))

    return gdf


def generate_time(start, end, n, ci=99):
    """
    Returns n request times with normal distribution st  ci% of requests falls between the start and end limit.
    :param start: lower bound
    :param end: upper bound
    :param n: sample size
    :param ci: confidence interval
    :return: np.array(n, 1)
    """
    h = 24*36e5
    start *= 36e5
    end *= 36e5
    mean = (end+start)/2
    std = compute_std(end, start, ci)
    times = np.random.normal(mean, std, n)
    times = np.round(times/1e3)*1e3
    times[times < 0] += h
    times[times >= h] -= h
    return times


def compute_std(end, start, ci)->float:
    """
    Computes standard deviation for the given values of
     mean, sample size, and confidence interval.

    :param mean: sample mean
    :param ci: confidence interval
    :return: standard deviation
    """
    zscore = {90: 1.645, 95: 1.96, 99: 2.576}
    std = (end-start)/zscore[ci]
    return std


def num_clusters(nodes: gpd.GeoDataFrame, crsm:int, cluster_size:float)->int:
    """

    :param nodes_proj: nodes with metric geometry
    :return: number of clusters for demand generation
    """
    nodes_proj = nodes.to_crs(crs=crsm)
    area_km = nodes_proj.unary_union.convex_hull.area/1e6
    if area_km <= cluster_size*5:
        return 5
    n = int(area_km / cluster_size)
    return n



if __name__ == '__main__':
    num_args = len(sys.argv)
    if num_args == 1:
        generate_trips()
    else:
        print(sys.argv[1])
        generate_trips(sys.argv[1])




