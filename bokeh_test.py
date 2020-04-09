import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.collections import LineCollection
from shapely.geometry import Point, Polygon, LineString
from os import path, listdir, getcwd
import matplotlib.pyplot as plt
import json
from shapely.geometry import box
from bokeh.io import show, output_file
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter,
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider, Column,BooleanFilter)
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.core.properties import value
from bokeh import tile_providers

import logging

log = logging.getLogger('bokeh')
log.setLevel(logging.INFO)
log.info('start')

# print(tile_providers.get_provider('CARTODBPOSITRON'))

def read_map(city, dir):
    """
    Reads map data from roads folder into dataframes
    :param city: mny, bj5, ch1
    :param dir: path to roads directory
    :return: nodes: DataFrame       (node_id, x, y)
             edges: GeoDataFrame    (u, v, length)
    """
    with open(path.join(dir, city + '_nodes.js')) as f:
        lines = f.read().split('=')
        node_dict = json.loads(lines[1])
    nodes = pd.DataFrame([[k, *v] for k, v in node_dict.items()], columns=['node_id', 'x', 'y'])
    nodes = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs=4326)

    with open(path.join(dir, city + '_directed.edges')) as f:
        lines = f.read().split('\n')
    num_nodes, num_edges = [int(x) for x in lines[0].split(' ')]
    assert nodes.shape[0] == num_nodes
    print('Graph with %s nodes and %s edges' % (num_nodes, num_edges))

    edge_list = [[int(x) for x in line.split()] for line in lines[1:] if len(line) > 0]
    edges = pd.DataFrame(edge_list, columns=['u', 'v', 'length'])
    return nodes, edges


def read_instance(filename):
    """
    Reads trip data into dataframe
    :param filename: path to instance file
    :return: GeoDataFrame   (id, origin, dest, q, early, late)
    """
    with open(filename) as f:
        lines = f.read().split('\n')
    # inst_name = lines[0]
    header = [l.lower() for l in lines[5].split()]
    lines = lines[6:]
    # data = [(int(x) for x in line.split()) for line in lines ]
    # the instance has errors, one line had 7 values, for instances w/o errors the oneliner should be enough
    data = []
    for line in lines:
        vals = line.split()
        if len(vals) == 6:
            data.append([int(x) for x in vals])
    df = pd.DataFrame(data, columns=header)
    return df


def get_coordinates(nodes, ids):
    return np.array([nodes.iloc[ids].x, nodes.iloc[ids].y])


def plot_bokeh_map(trips, nodes):
    """
    Plots interactive map of trips in browser.
    :param df: pandas.DataFrame from cargo instance (id, origin, dest,..., early, late)
    :param nodes: DataFrame/GeoDtatFrame (node id, x, y...)
    :return:
    """
    df = trips[trips.dest >= 0].copy()
    df['dropoff_time'] = df['late']
    df['pickup_time'] = df['early']
    df['minute'] = df['early'] // 10

    nodes_proj = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs=4326)
    nodes_proj = nodes_proj.to_crs(crs=3857)
    print(nodes_proj.head())

    df['x1'] = df.origin.apply(lambda n: nodes_proj.iloc[n].geometry.x)
    df['y1'] = df.origin.apply(lambda n: nodes_proj.iloc[n].geometry.y)
    df['x2'] = df.dest.apply(lambda n: nodes_proj.iloc[n].geometry.x)
    df['y2'] = df.dest.apply(lambda n: nodes_proj.iloc[n].geometry.y)
    df['xx'] = df.apply(lambda r: [r.x1, r.x2], axis=1)
    df['yy'] = df.apply(lambda r: [r.y1, r.y2], axis=1)
    # print(df.head())
    # print(df.columns)

    source0 = ColumnDataSource(data=dict(id=df.id.values, min=df.minute.values,
                                         pickup_time=df.early.values, dropoff_time=df.late.values,
                                         pickup_node=df.origin.values, dropoff_node=df.dest.values,
                                         x1=df.x1.values, y1=df.y1.values,
                                         x2=df.x2.values, y2=df.y2.values,
                                         xx=df.xx.values, yy=df.yy.values))
    df1 = df[df['minute'] == 0].copy()
    print(df1.head())

    source1 = ColumnDataSource(data=dict(id=df1.id.values,
                                         pickup_time=df1.early.values, dropoff_time=df1.late.values,
                                         pickup_node=df1.origin.values, dropoff_node=df1.dest.values,
                                         x1=df1.x1.values, y1=df1.y1.values,
                                         x2=df1.x2.values, y2=df1.y2.values,
                                         xx=df1.xx.values, yy=df1.yy.values))
    # p = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
    #            x_axis_type="mercator", y_axis_type="mercator")

    plot = figure(plot_width=1200, plot_height=800, x_range=(-2000000, 600000), y_range=(-1000000, 7000000),
                  x_axis_type='mercator', y_axis_type='mercator')
    # all nodes
    # plot.circle('x1', 'y1', source=source0, line_alpha=0.4, size=2, color='black')
    # tiles
    tile_provider = tile_providers.get_provider('CARTODBPOSITRON')
    print(tile_provider)
    plot.add_tile(tile_provider)
    # trips
    pickups = plot.circle('x1', 'y1', source=source1, line_alpha=0.9, color='red', size=7)
    dropoffs = plot.circle('x2', 'y2', source=source1, line_alpha=0.9, color='green', size=7)
    routes = plot.multi_line(xs='xx', ys='yy', source=source1, color='blue', width=1.2)

    time_slider = Slider(start=0, end=1800, value=0, step=10, title="Time, s")

    callback = CustomJS(args=dict(source=source0, dest=source1, time=time_slider),
                        code="""
        dest.data.id = [];
        dest.data.x1 = [];
        dest.data.y1 = [];
        dest.data.x2 = [];
        dest.data.y2 = [];
        dest.data.xx = [];
        dest.data.yy = [];
        dest.data.pickup_time = [];
        dest.data.dropoff_time = [];
        dest.data.pickup_node = [];
        dest.data.dropoff_node = [];
        const t = time.value/10;
       // console.log("callback " +t);
       // console.log("source data "+source.get_length());
        for (var i = 0; i < source.get_length(); i++) {
              if(source.data.min[i] == t){
             //   console.log(i +" "+ source.data.x1[i] +" "+ source.data.y1[i]);
                dest.data.id.push(source.data.id[i]);
                dest.data.x1.push(source.data.x1[i]);
                dest.data.y1.push(source.data.y1[i]);
                dest.data.x2.push(source.data.x2[i]);
                dest.data.y2.push(source.data.y2[i]);
                dest.data.xx.push(source.data.xx[i]);
                dest.data.yy.push(source.data.yy[i]);
                dest.data.pickup_time.push(source.data.pickup_time[i]);
                dest.data.dropoff_time.push(source.data.dropoff_time[i]);
                dest.data.pickup_node.push(source.data.pickup_node[i]);
                dest.data.dropoff_node.push(source.data.dropoff_node[i]);

            }
        }
       // console.log(dest.data.x1.length + " " +dest.data.y1.length);
        dest.change.emit();
    """)
    time_slider.js_on_change('value', callback)

    #
    plot.add_tools(HoverTool(renderers=[pickups],
                             tooltips=[('trip id', '@id'),
                                       ('pickup time', '@pickup_time'),
                                       ('dropoff time', '@dropoff_time'),
                                       ('pickup node', '@pickup_node')]))

    plot.add_tools(HoverTool(renderers=[dropoffs],
                             tooltips=[('trip id', '@id'),
                                       ('pickup time', '@pickup_time'),
                                       ('dropoff time', '@dropoff_time'),
                                       ('dropoff node', '@dropoff_node')]))

    plot.add_tools(HoverTool(renderers=[routes],
                             tooltips=[('trip id', '@id'),
                                       ('pickup time', '@pickup_time'),
                                       ('dropoff time', '@dropoff_time'),
                                       ('pickup node', '@pickup_node'),
                                       ('dropoff node', '@dropoff_node')]))

    #
    layout = column(plot, row(time_slider))
    output_file("slider.html", title="slider.py example")

    show(layout)


ROADS = path.join(getcwd(), 'cargo', 'roads')
city = 'mny'
ny_nodes, ny_edges = read_map(city, ROADS)
# print(nodes.head())

 ## 5033 customers
inst = 'rs-mny-m10k-c3-d6-s10-x1.0.instance'
ny_trips = read_instance(path.join(getcwd(), 'cargo', 'instances', inst))

plot_bokeh_map(ny_trips, ny_nodes)