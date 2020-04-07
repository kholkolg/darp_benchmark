import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.ckdtree import cKDTree

from plots import plot_trip_counts, plot_histogram


def prepare_trips(filename, owner:str, geoname: str, crs: int)->gpd.GeoDataFrame:
    if owner == 'liftago':
        return liftago(filename, geoname, crs)

    if owner == 'tlc':
        return tlc(filename, geoname, crs)


def tlc(filename:str, geoname: str, crs: int)->gpd.GeoDataFrame:
    cols = ['tpep_pickup_datetime', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

    df = pd.read_csv(filename, index_col=None, header=0, sep=',', parse_dates=True)
    df = df[cols].rename(columns={'tpep_pickup_datetime':'pickup_datetime'})


    df = df[(df['pickup_datetime'] >='2015-08-03') & (df['pickup_datetime'] < '2015-08-04')]
    df['hour'] = pd.DatetimeIndex(df['pickup_datetime']).hour
    # print(df.head())


    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.pickup_longitude, df.pickup_latitude))
    gdf = gdf.rename(columns={'geometry': 'p_' + geoname}).set_geometry('p_' + geoname)
    gdf['d_' + geoname] = gpd.points_from_xy(df.dropoff_longitude, df.dropoff_latitude)
    gdf.crs = crs

    return gdf


def liftago(filename:str, geoname: str, crs: int)->gpd.GeoDataFrame:
    # parse original txt file containing  286009 requests.
    # request time (ms from 00:00), pickup latitude, pickup longitude, dropoff latitude, dropoff longitude
    # separated by space
    cols = ['time', 'pickup_lat', 'pickup_lon', 'dropoff_lat', 'dropoff_lon']

    df = pd.read_csv(filename, index_col=None, header=None, sep=' ', names=cols)
    df.rename(columns={'time': 'time_ms'}, inplace=True)
    df.sort_values(by='time_ms', inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['hour'] = np.round(df['time_ms']/36e5)
    # df['time'] = pd.to_datetime(df['time_ms'], unit='ms')

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.pickup_lon, df.pickup_lat))
    gdf = gdf.rename(columns={'geometry': 'p_'+geoname}).set_geometry('p_'+geoname)
    gdf['d_'+geoname] = gpd.points_from_xy(df.dropoff_lon, df.dropoff_lat)
    gdf.crs = crs

    return gdf


def trips_by_hour(trips: gpd.GeoDataFrame, plot_flinename: str)->pd.DataFrame:
    # number of trips by hour
    if 'hour' not in trips.columns:
        trips['hour'] = np.round(trips.time_ms/36e5)
    counts = trips.groupby(['hour']).size()
    print(counts.head(20))
    counts = pd.DataFrame(counts, columns=['trip_count'])
    total = np.sum(counts.trip_count.values)
    counts['percent_of_total'] = np.round(100*counts.trip_count / total, decimals=2)

    if plot_flinename:
        plot_trip_counts(counts, 'trip_count',  plot_flinename)

    return counts


def map_od_to_nodes(nodes: gpd.GeoDataFrame, trips: gpd.GeoDataFrame, geoname: str,
                    crs: str) ->gpd.GeoDataFrame:

    tree = cKDTree(np.array(list(zip(nodes[geoname].x, nodes[geoname].y))))

    #pickup nodes
    trips_ = trips.set_geometry('p_'+geoname, crs=crs)
    dist, idx = tree.query(np.array(list(zip(trips_['p_'+geoname].x, trips_['p_'+geoname].y))), k=1)

    p_gdf = pd.concat([trips_.reset_index(drop=True), nodes.loc[idx].reset_index(drop=True),
             pd.Series(dist, name='dist')], axis=1)
    p_gdf = p_gdf[['osmid', 'geometry', 'centroid_label']]
    p_gdf.rename(columns={'osmid': 'p_osmid', 'geometry': 'pn_'+geoname,
                                 'centroid_label': 'p_centroid'}, inplace=True)

    #dropoff nodes
    trips_.set_geometry('d_'+geoname, inplace=True, crs=crs)
    dist, idx = tree.query(np.array(list(zip(trips_['d_'+geoname].x, trips_['d_'+geoname].y))), k=1)

    d_gdf = pd.concat([trips_.reset_index(drop=True), nodes.loc[idx].reset_index(drop=True),
             pd.Series(dist, name='dist')], axis=1)
    d_gdf = d_gdf[['osmid', 'geometry', 'centroid_label']]
    d_gdf.rename(columns={'osmid': 'dropoff_node', 'geometry': 'pd_'+geoname,
                                 'centroid_label': 'd_centroid'}, inplace=True)

    trips_ = gpd.GeoDataFrame(pd.concat([trips_, p_gdf, d_gdf], axis=1), geometry='p_'+geoname)
    trips_.crs = crs
    return trips_


def trip_lengths(trips: gpd.GeoDataFrame, geoname: str, crs: int, plot_filename: str=None):
    crs0 = trips.crs
    trips_proj = trips.set_geometry('p_'+geoname).to_crs(epsg=crs)
    trips_proj = trips_proj.set_geometry('d_' + geoname, crs=crs0).to_crs(epsg=crs)

    trips_proj['trip_length'] = trips_proj.apply(lambda r: r['d_'+geoname].distance(r['p_'+geoname]), axis=1)
    # trips_proj = trips_proj[trips_proj.trip_length <= 25000]
    # print(trips_proj[['p_'+geoname, 'd_'+geoname, 'trip_length']].head())
    # trips_proj = trips_proj[trips_proj['trip_length'] <= 5]
    print(np.mean(trips_proj.trip_length.values), np.std(trips_proj.trip_length.values))
    print(np.min(trips_proj.trip_length.values), np.max(trips_proj.trip_length.values))
    if plot_filename:
        plot_histogram(trips_proj.trip_length.values, 50, 'trip length,[m]', 'Trip count', plot_filename)

    trips_proj = trips_proj.to_crs(crs0).set_geometry('p_'+geoname, crs=crs).to_crs(crs0)
    len0 = sum(trips_proj.trip_length <= 10)
    print('shorter than 10m : ', len0)
    return trips_proj



def project_all_geometries(gdf:gpd.GeoDataFrame, geoname: str, crs: int)->gpd.GeoDataFrame:
    """
    Projects all geometry columns contained in the dataframe to the given crs.

    :param gdf: data with one or more geometry columns
    :param geoname: part of the geometry column name same for all columns
    :param crs: crs to project to
    :return: same gdf with projected geometries
    """
    crs0 = gdf.crs
    geom_colums = [c for c in gdf.columns if geoname in c]
    gdf_proj = gdf.set_geometry(geom_colums[0], crs=crs0).to_crs(epsg=crs)
    for col in geom_colums[1:]:
        gdf_proj = gdf.set_geometry(col, crs=crs0).to_crs(epsg=crs)

    return gdf_proj
