#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from os import getcwd, path
from plots import plot_histogram, plot_histograms, plot_od_heatmap, plot_heatmap
from demand_utils import prepare_trips
from graph_utils import get_map_data
from typing import Tuple


OUTDIR = 'output'
PICDIR = 'pics'
INIT =  'epsg:4326'
GEO = 4326
METRIC = 3310
GNAME = 'geometry'


def count_by_cluster(trips: gpd.GeoDataFrame, centroids: gpd.GeoDataFrame, geoname: str,
                     crs: str, plot_filename:str=None)->pd.DataFrame:
    # Pickups and dropoff in and between clusters counts by hour
    # Count trips between pairs of clusters
    trips_clust = trips[['time_ms', 'p_centroid', 'd_centroid']]
    counts = trips_clust.groupby(['p_centroid', 'd_centroid']).size().reset_index(name='trip_count')
    counts = pd.merge(counts, centroids, left_on='p_centroid', right_on='label')
    counts.rename(columns={'x': 'p_x', 'y': 'p_y', 'from_center':' p_center', geoname: 'p_'+geoname}, inplace=True)
    counts.drop(columns='label', inplace=True)
    counts = pd.merge(counts, centroids, left_on='d_centroid', right_on='label')
    counts.rename(columns={'x': 'd_x', 'y': 'd_y', 'from_center': 'd_center', geoname: 'd_'+geoname}, inplace=True)
    counts.drop(columns='label', inplace=True)
    counts.crs = crs

    if plot_filename:
        # counts.hist(by='count', bins=50, figsize=(16, 12))
        plot_histogram(counts.trip_count, 50, 'Number of trips between a pair of clusters',
                       'Cluster count', plot_filename)
    return counts


def counts_inside_cluster(counts:pd.DataFrame, plot=True)->Tuple[pd.DataFrame, pd.DataFrame]:

    # Inside one cluster
    incluster = counts[counts.p_centroid == counts.d_centroid]
    incluster.drop(columns=['d_centroid', 'd_x', 'd_y', 'd_center','d_'+GNAME], inplace=True)
    incluster.rename(columns={'p_centroid': 'centroid', 'p_x': 'x', 'p_y': 'y',
                                     'p_'+GNAME: GNAME, 'p_center': 'center'}, inplace=True)
    # between the clusters
    btwcluster = counts[counts.p_centroid != counts.d_centroid]
    # if plot:
    #     fig, axes = plt.subplots(ncols=2, figsize=(16,12))
    #     colors = incluster.trip_count.values
    #     min_max_scaler = preprocessing.MinMaxScaler()
    #     colors_scaled = min_max_scaler.fit_transform(colors.reshape(-1, 1))
    #     colors_scaled = colors_scaled.reshape(colors.shape[0])
    #     cmap = [colors_scaled[i] for i in kmeans.labels_]
    #
    #     ax = axes[0]
    #     fig, ax = plt.subplots(figsize=(16, 16))
    #     ax.scatter([x for x, _ in data], [y for _, y in data], c=cmap, s=3, zorder=1)
    #
    #     ax.scatter([x for x, _ in centroids], [y for _, y in centroids], c='red', s=50, zorder=2)
    return incluster, btwcluster


def count_pickups_and_dropoffs_by_cluster(trips: pd.DataFrame, centroids: gpd.GeoDataFrame,
                                          plot_filename: str = None)->pd.DataFrame:
#pickup  and dropoff counts
    trips_clust = trips[['time_ms', 'hour', 'p_centroid', 'd_centroid']]

    p_counts = trips_clust.groupby('p_centroid').size()
    print(p_counts.head(10))
    # p_counts = pd.merge(p_counts, centroids, left_on='p_centroid', right_on='label')

    d_counts = trips_clust.groupby('d_centroid').size()
    print(d_counts.head(10))
    # d_counts = pd.merge(d_counts, centroids, left_on='d_centroid', right_on='label')
    if plot_filename:
        plot_histograms(p_counts.pickup_count.values, d_counts.dropoff_count.values, 50,
                        'Number of pickups in the cluster','Number of dropoffs in the cluster',
                        plot_filename)

    d_counts = d_counts[[c for c in d_counts.columns if c not in p_counts.columns]]
    counts = pd.merge(p_counts, d_counts, left_on='p_centroid', right_on='d_centroid')
    counts.rename(columns={'p_centroid':'label'}, inplace=True)

    return counts


def od_clusters_hourly(trips, centroids):
    #Origins by clusters, hour by hour
    trips_clust= trips[['time_ms', 'hour', 'p_centroid', 'd_centroid']]
    counts = trips_clust.groupby(['p_centroid', 'd_centroid', 'hour']).size().reset_index(name='count')
    counts = pd.merge(counts, centroids, left_on='p_centroid', right_on='label')
    counts.rename(columns={'x': 'p_x', 'y': 'p_y', 'from_center': 'p_center', GNAME: 'p_'+GNAME}, inplace=True)
    counts.drop(columns='label', inplace=True)
    counts = pd.merge(counts, centroids, left_on='d_centroid', right_on='label')
    counts.rename(columns={'x': 'd_x', 'y': 'd_y', 'from_center': 'd_center',GNAME: 'd_'+GNAME}, inplace=True)
    counts.drop(columns='label', inplace=True)
    counts.crs = 'epsg:4326'

    return counts




if __name__ == '__main__':

    # csv_pat = path.join(str(Path.home()), 'Downloads', 'trips.txt')
    city = 'Prague'  # 20828 nodes, 48397 edges

    nodes, edges = get_map_data(city, GEO)
    centroids = cluster_points(nodes, 50, GNAME, GEO, METRIC)
    # edges.to_file(driver='ESRI Shapefile', filename="liftago/map.shp")

    # print(nodes.head())
    # print(centroids.head())

    lft_path = path.join(str(Path.home()), 'Downloads', 'trips.txt')
    trips = prepare_trips(lft_path, 'liftago', GNAME, GEO)
    print(trips.head())

    # fig, axes = plt.subplots(ncols=2, figsize=(20, 12))
    # plot_heatmap(trips.pickup_lon, trips.pickup_lat, axes[0])
    # plot_heatmap(trips.dropoff_lon, trips.dropoff_lat, axes[1])
    # plt.savefig('od_heatmap_prague.png')
    # plt.show()
    # ALL TRIPS
    # pdf = trips[['time_ms', 'hour', 'p_geometry']]
    # pdf = pdf.set_geometry('p_geometry', crs=4326)
    # pdf.to_file(driver='ESRI Shapefile', filename="liftago/trips_pickup.shp")
    #
    # ddf = trips[['time_ms', 'hour', 'd_geometry']]
    # ddf = ddf.set_geometry('d_geometry', crs=4326)
    # ddf.to_file(driver='ESRI Shapefile', filename="liftago/trips_dropoff.shp")

    # MORNING RUSH HOURS
    # trips1 = trips[(trips.hour >= 5) & (trips.hour < 8)]
    # pdf = trips1[['time_ms', 'hour', 'p_geometry']]
    # pdf = pdf.set_geometry('p_geometry', crs=4326)
    # pdf.to_file(driver='ESRI Shapefile', filename="liftago/morning_pickup.shp")
    #
    # ddf = trips1[['time_ms', 'hour', 'd_geometry']]
    # ddf = ddf.set_geometry('d_geometry', crs=4326)
    # ddf.to_file(driver='ESRI Shapefile', filename="liftago/morning_dropoff.shp")
    #
    # # EVENING RUSH HOURS
    # trips1 = trips[(trips.hour >= 15) & (trips.hour < 18)]
    # pdf = trips1[['time_ms', 'hour', 'p_geometry']]
    # pdf = pdf.set_geometry('p_geometry', crs=4326)
    # pdf.to_file(driver='ESRI Shapefile', filename="liftago/evening_pickup.shp")
    #
    # ddf = trips1[['time_ms', 'hour', 'd_geometry']]
    # ddf = ddf.set_geometry('d_geometry', crs=4326)
    # ddf.to_file(driver='ESRI Shapefile', filename="liftago/evening_dropoff.shp")


    #
    # trips = map_od_to_nodes(nodes, trips, GNAME, INIT)
    # trips = trip_lengths(trips, GNAME, METRIC, path.join(getcwd(), PICDIR, city+'_trip_lengths_hist.png'))
    # print(trips.head())
    #mean = 6467.321217792287, std 3729.5241584457826
    # 10199.386420263034
    # 5010.515464108012
    # counts = count_by_cluster(trips, centroids, GNAME, INIT, path.join(getcwd(), PICDIR, 'cluster_hist.png'))

    # counts = count_pickups_and_dropoffs_by_cluster(trips, centroids,
    #                                                path.join(getcwd(), PICDIR, 'pick_drop_hists.png'))
    # print(counts.head())
    #path.join(getcwd(), PICDIR, 'cluster_pd_hist.png')






# #trip lengths\
# def sp_length(us, vs):
#     lens = []
#     for u, v in zip(us, vs):
#         id = G['osmid'][u]
#         print(id)
#         lens.append(nx.shortest_path_length(G, u, v, weight='length'))
#     return lens
#
#
#
# new_df['trip_len']  = new_df.apply(lambda x: sp_length(nodes_df.osmid.iloc[new_df.pickup_node.values],
#                                                        nodes_df.osmid.iloc[new_df.dropoff_node.values]), axis=1)



# trips_proj.set_geometry('pgeom', inplace=True)
# trips_proj['pgeom2']= trips_proj.pgeom.apply(lambda x: center_point(x))
# trips_proj['dgeom2']= trips_proj.dgeom.apply(lambda x: center_point(x))
# print(trips_proj.head())
# trip_vec = trips_proj[['time_ms','pgeom2', 'dgeom2']]
# trip_vec['vx'] = trip_vec.apply(lambda r: r.dgeom2.x - r.pgeom2.x, axis=1)
# trip_vec['vy'] = trip_vec.apply(lambda r: r.dgeom2.y - r.pgeom2.y, axis=1)
# # trip_vec['vx'] = trip_vec['dgeom2'].y - trip_vec['pgeom2'].y
# print(trip_vec.head())
# trip_vec['norm'] = np.sqrt(trip_vec['vx']**2 +trip_vec['vy']**2)
# trip_vec['vx'] = trip_vec['vx']/trip_vec['norm']
# trip_vec['vy'] = trip_vec['vy']/trip_vec['norm']
# # quiver([X, Y], U, V, [C], **kw)
# vectors = [[np.array(trip_vec.pgeom2.apply(lambda p: [p.x, p.y]).values),
#            np.array(trip_vec.vx.values), np.array(trip_vec.vy.values)]]
# print(vectors[:][:4])
# fig, ax = plt.subplots()
# plt.quiver(*vectors[:10])
# plt.show()





