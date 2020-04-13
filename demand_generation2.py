import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
from os import path, getcwd, mkdir
from scipy.stats import sem, t
from shapely.geometry import Point, box
from sklearn.cluster import KMeans
from demand_utils import trips_by_hour, trip_lengths
from utils import load_nodes_gdf
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import  Rotation as R
from scipy.spatial import cKDTree






def generate_instance(nodes: gpd.GeoDataFrame, centroids: pd.DataFrame, min_dist:float,
                      avg_dist:float, num_requests: int, outputdir: str, rush_hours:List[Tuple[float, float]] = None):

    if rush_hours is None or len(rush_hours) == 0:
        print('no peaks')

        trips_df = pd.DataFrame(columns=['time_ms', 'cluster1', 'cluster2', 'dist', 'dx', 'dy', 'origin', 'dest'])

        trips_df['time_ms'] = generate_time(4, 20, num_requests, 95)

        trips_df['cluster1'] = select_clusters(centroids.label.values, num_requests, np.exp(centroids.node_count/1000))
        trips_df['cluster2'] = select_clusters(centroids.label.values, num_requests, np.exp(centroids.node_count/1000))

        trips_df = cluster_to_vector(trips_df, centroids)

        trips_df['dist'] = np.random.normal(avg_dist, avg_dist/2, size=num_requests)
        trips_df.loc[(trips_df.dist < min_dist), 'dist'] = min_dist
        # print(trips_df.head())
        trips_df = select_nodes(trips_df, nodes)
        save_trips_csv(trips_df, out_dir)
        return trips_df

    # else:
    #     print('peaks ', rush_hours)
    #     num_peaks = len(rush_hours)
    #     num_requests_n = int(0.6*num_requests)
    #     num_requests_p = num_requests - num_requests_n
    #
    #     times = generate_time(4, 20, num_requests_n, 95)
    #     # times = generate_time_uniform(0, 24, num_requests_n)
    #     # print(np.exp(centroids.node_count/100))
    #     cluster_ids = select_clusters(centroids.label.values, 2 * num_requests_n,
    #                                   np.multiply(np.random.uniform(0.1, 0.9, centroids.shape[0]), np.exp(centroids.node_count/100)))
    #     points = clusters_to_points(cluster_ids, nodes)
    #     points = points.reshape(num_requests_n, 4)
    #     df = np.append(times, points, axis=1)
    #     df = pd.DataFrame(df, columns=['time_ms', 'p_x', 'p_y', 'd_x', 'd_y'])
    #
    #     n = num_requests_p // num_peaks
    #     print('n ', n)
    #     for start, end in rush_hours:
    #         print('rush from ', start, ' to ', end)
    #         times = generate_time(start, end, n, 95)
    #         # probs = [centroids.from_center.values**2,
    #         #          1/centroids.from_center.values**2]
    #         probs = [centroids.from_center, 1/centroids.from_center]
    #         probs_p = probs[0] if start <= 12 else probs[1]
    #         probs_d = probs[1] if start <= 12 else probs[0]
    #         p_cluster_ids = select_clusters(centroids.label.values, n, probs_p)
    #         p_points = clusters_to_points(p_cluster_ids, nodes)
    #
    #         d_cluster_ids = select_clusters(centroids.label.values, n, probs_d)
    #         d_points = clusters_to_points(d_cluster_ids, nodes)
    #         # print(times.shape, p_points.shape, d_points.shape)
    #         new_df = np.append(times, p_points, axis=1)
    #         new_df = np.append(new_df, d_points, axis=1)
    #         print('new df ', new_df.shape)
    #         new_df = pd.DataFrame(new_df, columns=['time_ms', 'p_x', 'p_y', 'd_x', 'd_y'])
    #         df = pd.concat([df, new_df], axis=0)
    #         print('df ', df.shape)
    #
    #     # print(df.head())
    #     df = df.sort_values(by='time_ms').reset_index(drop=True)
    #
    #
    #     df.to_csv(path.join(outputdir, 'trips.csv'))
    #     return df


def cluster_to_vector(trips, centroids):
    trips['dx'] = centroids.iloc[trips.cluster2].x.values - centroids.iloc[trips.cluster1].x.values
    trips['dy'] = centroids.iloc[trips.cluster2].y.values - centroids.iloc[trips.cluster1].y.values
    n = sum(trips.dx == 0)
    trips.loc[(trips.dx == 0), 'dx'] = np.random.uniform(-1, 1, size=n)
    trips.loc[(trips.dy == 0), 'dy'] = np.random.uniform(-1, 1, size=n)
    trips['norm'] = np.sqrt(trips.dx ** 2 + trips.dy ** 2)
    trips['dx'] = trips['dx'] / trips['norm']
    trips['dy'] = trips['dy'] / trips['norm']
    trips = trips.drop(columns=['norm'])
    return trips


def select_nodes(trips, nodes):

    p_clusters = trips.cluster1.unique()
    for c in p_clusters:
        cluster_nodes = nodes[nodes.centroid_label == c].index
        num = sum(trips.cluster1 == c)
        trips.loc[(trips.cluster1 == c), 'origin'] = np.random.choice(cluster_nodes, size=num, replace=True)

    # print(trips.head())
    trips['x2'] = nodes.iloc[trips['origin']].geometry.x.values + trips.dx*trips.dist
    trips['y2'] = nodes.iloc[trips['origin']].geometry.y.values + trips.dy * trips.dist

    # print(trips.head())

    tree = cKDTree(np.array((nodes.geometry.x, nodes.geometry.y)).T)
    dist, idx = tree.query(np.array((trips.x2, trips.y2)).T, k=1)
    trips['dest'] = idx
    trips = trips.drop(columns=['x2', 'y2'])
    # print(trips.head())
    return trips


def save_trips_csv(trips, dir):
    df = trips[['time_ms', 'origin', 'dest']]
    df.to_csv(path.join(dir, 'trips.csv'), sep='\t')


def save_shapefiles(trips, nodes, dir, pcol='origin', dcol='dest'):

    crs=4326
    nodes_ = nodes.to_crs(crs=crs)

    pickup = trips[['time_ms', pcol]]
    pickup['geometry'] = nodes_.iloc[pickup[pcol]].geometry
    pickup = gpd.GeoDataFrame(pickup, geometry='geometry', crs=4326)
    pickup.to_file(driver='ESRI Shapefile', filename=path.join(dir, "pickup.shp"))

    dropoff = trips[['time_ms', dcol]]
    dropoff['geometry'] = nodes_.iloc[dropoff[dcol]].geometry
    dropoff = gpd.GeoDataFrame(pickup, geometry='geometry', crs=crs)
    dropoff.to_file(driver='ESRI Shapefile', filename=path.join(dir, "dropoff.shp"))


def select_clusters(points, num_points, probs=None):

    probs = probs if probs is not None else np.ones(len(points))
    if np.sum(probs) != 1:
        probs = probs/np.sum(probs)
    result = np.random.choice(points, size=num_points, p=probs)
    return result


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
    # print('mean ', mean/36e5, ', std ', std/36e5)
    times = np.random.normal(mean, std, n)
    times = np.round(times/1e3)*1e3
    times[times < 0] += h
    times[times >= h] -= h
    # times = times.reshape(-1, 1)
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
    #h = t.ppf((1 + ci/100)/2, n - 1)
    std = (end-start)/zscore[ci]
    return std


def num_clusters(nodes_proj:gpd.GeoDataFrame):
    area_km = nodes_proj.unary_union.convex_hull.area/1e6
    if area_km <= 150:
        return 5
    n = int(area_km/30)
    return n


def cluster_points2(points: gpd.GeoDataFrame, num_clusters: int, plot_filename=None) -> pd.DataFrame:
    crs = points.crs
    print('cluster points ', crs)
    coords = np.array([points.geometry.x, points.geometry.y]).T
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coords)
    centroids = kmeans.cluster_centers_

    points['centroid_label'] = kmeans.labels_
    centroids_df = pd.DataFrame(np.array([range(len(centroids)), centroids.T[0], centroids.T[1]]).T,
                                columns=['label', 'x', 'y'])
    centroids_df['label'] = centroids_df.label.apply(int)

    centroids = gpd.GeoDataFrame(centroids_df, geometry=gpd.points_from_xy(centroids_df.x, centroids_df.y),
                                 crs=crs)
    city_center = box(*points.unary_union.bounds).centroid
    centroids['from_center'] = centroids.geometry.apply(lambda point: point.distance(city_center))
    centroids['node_count'] = centroids.label.apply(lambda label: len([l for l in kmeans.labels_ if l == label]))

    return centroids


if __name__ == '__main__':
    city = 'Prague'
    dir = getcwd()
    inp_dir = path.join(dir, 'input')
    pic_dir = path.join(dir, 'pics')
    out_dir = path.join(dir, 'length', city+'_10k')
    try:
        mkdir(out_dir)
    except FileExistsError:
        print(out_dir, ' already exists')

    geo_name = 'geometry'
    crs0 = 4326
    crs1 = 3310

    nodes = load_nodes_gdf(city, geo_name, crs0, inp_dir)
    nodes = nodes.to_crs(crs=crs1)

    n_clusters = num_clusters(nodes)
    print('Clusters: ', n_clusters)
    centroids = cluster_points2(nodes, n_clusters, geo_name)
    print(nodes.head())
    print(centroids.head())
    trips = generate_instance(nodes, centroids, min_dist=100, avg_dist=5000,
                              num_requests=10000, outputdir=out_dir)
    print(trips.head())
    save_shapefiles(trips, nodes,  out_dir)
    # #
    trips['p_geometry'] = nodes.iloc[trips['origin']].geometry.values
    trips['d_geometry'] = nodes.iloc[trips['dest']].geometry.values
    trips_geo = gpd.GeoDataFrame(trips, geometry='p_geometry', crs=nodes.crs)


    trips_by_hour(trips_geo, path.join(out_dir, 'hourly_counts.png') )
    trip_lengths(trips_geo, geo_name, crs1, path.join(out_dir, 'length_histogram.png'))
    # # # print(trips.head())




    # # trips_geo['hour'] = np.round(trips_geo['time_ms']/36e5)
    # # MORNING RUSH HOURS
    # trips1 = trips_geo[(trips_geo.hour >= 5) & (trips_geo.hour < 8)]
    # pdf = trips1[['time_ms', 'hour', 'p_geometry']]
    # pdf = pdf.set_geometry('p_geometry', crs=4326)
    # pdf.to_file(driver='ESRI Shapefile', filename=path.join(out_dir, 'morning_pickup.shp'))
    #
    # ddf = trips1[['time_ms', 'hour', 'd_geometry']]
    # ddf = ddf.set_geometry('d_geometry', crs=4326)
    # ddf.to_file(driver='ESRI Shapefile', filename=path.join(out_dir, "morning_dropoff.shp"))
    #
    # # EVENING RUSH HOURS
    # trips1 = trips_geo[(trips_geo.hour >= 15) & (trips_geo.hour < 18)]
    # pdf = trips1[['time_ms', 'hour', 'p_geometry']]
    # pdf = pdf.set_geometry('p_geometry', crs=4326)
    # pdf.to_file(driver='ESRI Shapefile', filename=path.join(out_dir, "evening_pickup.shp"))
    #
    # ddf = trips1[['time_ms', 'hour', 'd_geometry']]
    # ddf = ddf.set_geometry('d_geometry', crs=4326)
    # ddf.to_file(driver='ESRI Shapefile', filename=path.join(out_dir, "evening_dropoff.shp"))

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


# def rotate(row):
#     angle = np.
#     cos_ = np.cos(row.angle)
#     sin_ = np.sin(row.angle)
#     return np.array((row.dx, row.dy)).T @  np.array([[cos_, -sin_], [sin_, cos_]])