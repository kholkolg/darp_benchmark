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
from bokeh.palettes import brewer
from bokeh.plotting import figure
from bokeh.core.properties import value


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


def add_geometry_column(df, nodes, column_name, geoname):
    """

    :param df: dataframe
    :param nodes: geodataframe
    :param column_name: name of column
    :param geoname: name of geometry column
    :return:
    """
    df['x'] = df[column_name].apply(lambda n: nodes.iloc[n].x)
    df['y'] = df[column_name].apply(lambda n: nodes.iloc[n].y)
    geoname = '_'.join([column_name, geoname])
    df[geoname] = gpd.points_from_xy(df.x, df.y)
    df = df.drop(columns=['x', 'y'])
    return df


def df_to_gdf(df, nodes, columns, geoname='geometry'):
    """
    Transforms columns from the list to geometry columns 'column_geometry',
    'origin' -> 'origin_geometry'.
    Values in the given columns are node_ids from nodes dataframe.

    :param df: DataFrame(col1, col2, col3...)
    :param nodes: GeoDataFrame (node_id, x, y)
    :param columns: list[col1, col3 ...]
    :param geoname: str
    :return: GeoDataFrame (col1_geometry, col2_geometry, ...)
    """
    df_ = df[:][:]
    for col in columns:
        df_ = add_geometry_column(df_, nodes, col, geoname)

    geoname = '_'.join([columns[0], geoname])
    gdf = gpd.GeoDataFrame(df_, geometry=geoname, crs=4326)
    return gdf

# ROADS = path.join(getcwd(), 'cargo', 'roads')
# city = 'mny'
# nodes, edges = read_map(city, ROADS)
# print(nodes.head())
#
#  ## 5033 customers
# inst = 'rs-mny-m10k-c3-d6-s10-x1.0.instance'
# trips = read_instance(path.join(getcwd(), 'instances', inst))
#
# df = trips[trips.dest >= 0]
# df['minute'] = df['early']//60

x = np.linspace(0, 10, 500)
y = np.sin(x)

source = ColumnDataSource(data=dict(x=x, y=y))

plot = figure(y_range=(-10, 10), plot_width=400, plot_height=400)

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

amp_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Amplitude")
freq_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Frequency")
phase_slider = Slider(start=0, end=6.4, value=0, step=.1, title="Phase")
offset_slider = Slider(start=-5, end=5, value=0, step=.1, title="Offset")

callback = CustomJS(args=dict(source=source, amp=amp_slider, freq=freq_slider, phase=phase_slider, offset=offset_slider),
                    code="""
    const data = source.data;
    const A = amp.value;
    const k = freq.value;
    const phi = phase.value;
    const B = offset.value;
    const x = data['x']
    const y = data['y']
    for (var i = 0; i < x.length; i++) {
        y[i] = B + A*Math.sin(k*x[i]+phi);
    }
    source.change.emit();
""")

amp_slider.js_on_change('value', callback)
freq_slider.js_on_change('value', callback)
phase_slider.js_on_change('value', callback)
offset_slider.js_on_change('value', callback)

layout = row(
    plot,
    column(amp_slider, freq_slider, phase_slider, offset_slider),
)

output_file("slider.html", title="slider.py example")

show(layout)