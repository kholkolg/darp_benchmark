import configparser
from os import path, mkdir

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd



def load_nodes_gdf(place: str, crsg:int, input_dir: str) -> gpd.GeoDataFrame:

    try:
        nodes = pd.read_csv(path.join(input_dir, place+'_demand_nodes.csv'), index_col=None)
        nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs=crsg)
        return nodes

    except:
        try:
            G = ox.graph_from_place(place, network_type='drive', simplify=True)
            G = nx.relabel.convert_node_labels_to_integers(G)
            nodes, edges = ox.graph_to_gdfs(G)
            nodes = nodes.reset_index().rename(columns={'index': 'id'})
            nodes = nodes[['id', 'x', 'y', 'osmid', 'geometry', 'highway']]

            edges = edges[['u', 'v', 'length', 'highway']]
            add_node_highway_tags(nodes, G)
            nodes.to_csv(path.join(input_dir,  place+'_demand_nodes.csv'))
            save_map_csv(nodes, edges, place, input_dir)
            return nodes

        except:
            print('Failed to load map data.')
            exit(-1)


def add_node_highway_tags(nodes, G):
    for u, v, d in G.edges(data=True):
        if 'highway' in d.keys():
            tag = d['highway']
            tag = tag[0] if isinstance(tag, list) else tag
            nodes.loc[nodes.index[[u]], 'highway'] = tag
            nodes.loc[nodes.index[[v]], 'highway'] = tag


def save_trips_csv(trips, dir):
    df = trips[['time_ms', 'origin', 'dest']]
    df.to_csv(path.join(dir, 'trips.csv'), sep='\t', index=False)


def save_map_csv(nodes, edges, place, dir):
    df1 = nodes[['id', 'osmid', 'x', 'y']]
    df1.to_csv(path.join(dir, place.lower()+'_nodes.csv'), sep='\t', index=False)

    df2 = edges[['u', 'v', 'length']]
    df2.to_csv(path.join(dir, place.lower() + '_edges.csv'), sep='\t', index=False)


def save_shapefiles(trips, nodes, crsg, dir):
    nodes_ = nodes.to_crs(crs=crsg)
    pickup = trips[['time_ms', 'origin']].copy()

    pickup['geometry'] = nodes_.iloc[pickup['origin']].geometry.values
    pickup = gpd.GeoDataFrame(pickup, geometry='geometry', crs=crsg)
    try:
        mkdir(path.join(dir, 'shapefiles'))
    except:
        pass
    pickup.to_file(driver='ESRI Shapefile', filename=path.join(dir, 'shapefiles', 'pickup.shp'))

    dropoff = trips[['time_ms', 'dest']].copy()
    dropoff['geometry'] = nodes_.iloc[dropoff['dest']].geometry.values
    dropoff = gpd.GeoDataFrame(dropoff, geometry='geometry', crs=crsg)
    dropoff.to_file(driver='ESRI Shapefile', filename=path.join(dir, 'shapefiles', 'dropoff.shp'))


def prepare_config(case_name:str):
    config = configparser.ConfigParser()
    config.read('config.ini')
    try:
        params = config[case_name]
    except:
        print('Config failed, default parameters are used')
        params = config['DEFAULT']

    params['outputDir'] = path.join(params['outputDir'], params['name'])
    try:
        mkdir(params['outputDir'])
    except FileExistsError:
        pass
    return params