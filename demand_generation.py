import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from os import path, getcwd, mkdir
from scipy.stats import sem, t

from demand_utils import trips_by_hour, trip_lengths
from utils import load_nodes_gdf, cluster_points
from typing import List, Tuple
import matplotlib.pyplot as plt




def generate_instance(nodes: gpd.GeoDataFrame, centroids: pd.DataFrame,
                      num_requests: int, outputdir: str, rush_hours:List[Tuple[float, float]] = None):

    if rush_hours is None or len(rush_hours) == 0:
        print('no peaks')
        times = generate_time(2, 22, num_requests, 90)
        cluster_ids = select_clusters(centroids.label.values, 2 * num_requests, centroids.node_count.values)
        points = clusters_to_points(cluster_ids, nodes)
        points = points.reshape(num_requests, 4)
        df = np.append(times, points, axis=1)
        df = pd.DataFrame(df, columns=['time_ms', 'p_x', 'p_y', 'd_x', 'd_y'])
        df.to_csv(path.join(outputdir, 'trips.csv'))
        return df

    else:
        print('peaks ', rush_hours)
        num_peaks = len(rush_hours)
        num_requests_n = num_requests//2
        num_requests_p = num_requests

        # times = generate_time(2, 22, num_requests_n, 90)
        times = generate_time_uniform(0, 24, num_requests_n)

        cluster_ids = select_clusters(centroids.label.values, 2 * num_requests_n, 2 * centroids.node_count.values)
        points = clusters_to_points(cluster_ids, nodes)
        points = points.reshape(num_requests_n, 4)
        df = np.append(times, points, axis=1)
        df = pd.DataFrame(df, columns=['time_ms', 'p_x', 'p_y', 'd_x', 'd_y'])

        n = num_requests_p // num_peaks
        for start, end in rush_hours:
            print('rush from ', start, ' to ', end)
            times = generate_time(start, end, n)
            probs = [np.multiply(2*centroids.node_count.values, centroids.from_center.values/1e3),
                     np.divide(2*centroids.node_count.values, centroids.from_center.values/1e3)]
            probs_p = probs[0] if start <= 12 else probs[1]
            probs_d = probs[1] if start <= 12 else probs[0]
            p_cluster_ids = select_clusters(centroids.label.values, n, 2 * centroids.node_count.values)
            p_points = clusters_to_points(p_cluster_ids, nodes)

            d_cluster_ids = select_clusters(centroids.label.values, n, 2 * centroids.node_count.values)
            d_points = clusters_to_points(d_cluster_ids, nodes)
            # print(times.shape, p_points.shape, d_points.shape)
            new_df = np.append(times, p_points, axis=1)
            new_df = np.append(new_df, d_points, axis=1)
            print('new df ', new_df.shape)
            new_df = pd.DataFrame(new_df, columns=['time_ms', 'p_x', 'p_y', 'd_x', 'd_y'])
            df = pd.concat([df, new_df], axis=0)
            print('df ', df.shape)

        print(df.head())
        df = df.sort_values(by='time_ms').reset_index(drop=True)
        df.to_csv(path.join(outputdir, 'trips.csv'))
        return df


def save_shapefiles(df, crs, dir):
    pdf = df[['time_ms', 'p_x', 'p_y']]
    pgdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf.p_x, pdf.p_y), crs=crs)
    print(pgdf.head())
    pgdf.to_file(driver='ESRI Shapefile', filename=path.join(dir,  "pickup.shp"))

    ddf = df[['time_ms', 'd_x', 'd_y']]
    dgdf = gpd.GeoDataFrame(ddf, geometry=gpd.points_from_xy(ddf.d_x, ddf.d_y), crs=crs)
    print(dgdf.head())
    dgdf.to_file(driver='ESRI Shapefile', filename=path.join(dir,  "dropoff.shp"))


def clusters_to_points(clusters, nodes):
    """
    Replaces all cluster labels with node coordinates from those clusters.

    :param clusters: sequence of cluster labels , shape = (n,)
    :param nodes: dataframe with nodes
    :return: node coordinates, np.array(n, 2)
    """
    points = []
    for cluster in clusters:
        cluster_nodes = nodes[nodes['centroid_label'] == cluster]
        node_ind = select_clusters(cluster_nodes.index.values, 1)
        point = nodes.iloc[node_ind][geo_name]
        coords = (point.x.values[0], point.y.values[0])
        points.append(coords)

    points = np.array(points)
    return points


def select_clusters(points, num_points, probs=None):

    probs = probs if probs is not None else np.ones(len(points))
    if np.sum(probs) != 1:
        probs = probs/np.sum(probs)
    result = np.random.choice(points, size=num_points, p=probs)
    return result


def generate_time_uniform(start, end, n):
    start *= 36e5
    end *= 36e5
    a = np.arange(start, end, 1e3)
    # probs = []
    # if np.sum(probs) != 1:
    #     probs = probs/np.sum(probs)
    times = np.random.choice(a, size=n)
    times = times.reshape(-1, 1)
    return times


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
    std = compute_std(end, start, n, ci)
    print('mean ', mean/36e5, ', std ', std/36e5)
    times = np.random.normal(mean, std, n)

    # times = times[times >= 0]
    # times = times[times < h]
    # times = times[:n]

    times[times < 0] += h
    times[times >= h] -= h
    times = np.round(times/1e3)*1e3
    times = times.reshape(-1, 1)
    return times


def compute_std(end, start, n, ci):
    """
    Computes standard deviation for the given values of
     mean, sample size, and confidence interval.

    :param mean: sample mean
    :param n: sapmle size
    :param ci: confidence interval
    :return: standard deviation
    """
    zscore = {90: 1.645, 95: 1.96, 99: 2.576}
    # h = t.ppf((1 + ci/100)/2, n - 1)
    # print('h=',h)
    std = (end-start)/(2*zscore[ci]) # np.sqrt(n)*(ulim-llim) / (2*zscore[ci])
    return std


if __name__ == '__main__':
    city = 'Prague'
    dir = getcwd()
    inp_dir = path.join(dir, 'input')
    pic_dir = path.join(dir, 'pics')
    out_dir = path.join(dir, 'output', city)
    try:
        mkdir(out_dir)
    except FileExistsError:
        print(out_dir, ' already exists')

    geo_name = 'geometry'
    crs0 = 4326
    crs1 = 3310

    nodes = load_nodes_gdf(city, geo_name, crs0, inp_dir)

    centroids = cluster_points(nodes, 50, geo_name, crs=crs0, metric_crs=crs1)

    trips = generate_instance(nodes, centroids, 10000, out_dir, [(5, 7), (15, 17)]) #
    save_shapefiles(trips, crs0, out_dir)
    trips_geo = gpd.GeoDataFrame(trips, geometry=gpd.points_from_xy(trips.p_x, trips.p_y), crs=crs0)
    trips_geo = trips_geo.rename(columns={geo_name: 'p_' + geo_name}).set_geometry('p_' + geo_name, crs=crs0)
    trips_geo['d_'+geo_name] = gpd.points_from_xy(trips_geo.d_x, trips_geo.d_y)

    trips_by_hour(trips_geo, path.join(out_dir, 'hourly_counts_u.png') )
    trip_lengths(trips_geo, geo_name, crs1, path.join(out_dir, 'length_histogram_u.png'))



#def generate_normal(nodes: gpd.GeoDataFrame, centroids:pd.DataFrame, num_requests:int, start:float, end:float,
#                     ):
#     # num_nodes = nodes.shape[0]
#     # print('num nodes ', num_nodes)
#     # start *= 36e5
#     # end *= 36e5
#     # mean = (end-start)/2
#     # std = compute_std(mean, num_nodes, 95)
#     # print('mean ', mean, ', std ', std)
#     # # times = generate_time(mean, std, num_requests)
#     # times = generate_time(mean, std, num_requests)
#     # print('generated mean ', np.mean(times), ', std ', np.std(times))
#     # # fig, ax = plt.subplots()
#     # # ax.hist(times, bins=24)
#     # # plt.show()
#
#     cluster_ids = choose_points(centroids.label.values, 2*num_requests, centroids.node_count.values)
#     # print(cluster_ids)
#
#     points = clusters_to_points(cluster_ids, nodes)
#     points = points.reshape(num_requests, 4)
#
#
#     # print(points.shape)
#     # times = times.reshape(-1, 1)
#     # print(times.shape)
#     df = np.append(times, points, axis=1)
#     # print(df.shape)
#     # print(df)
#     df = pd.DataFrame(df, columns=['time_ms', 'p_x', 'p_y', 'd_x', 'd_y'])
#     # print(df.head())
#
#     return df
