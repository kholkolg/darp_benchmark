import pandas as pd
import geopandas as gpd
from os import path, listdir, getcwd
import json
from bokeh.io import show, output_file
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh import tile_providers
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Slider, Column
import bs4
import logging


log = logging.getLogger('bokeh')
log.setLevel(logging.INFO)

WM = 3857 # web mercator for bokeh tiles
CITIES = {'mny': 'New York', 'bj5': 'Beijing', 'cd1': 'Chengdu'}
RANGES = {'mny': [(-8242000, -8210000), (4965000, 4990000)],
          'bj5': [(11600000, 11700000), (3900000, 4100000)], #TODO ranges for China
          'cd1': [(10080000, 11620000), (3660000, 3900000)]} #


def show_instance(input_path: str, inst_name: str, output_path: str=None):
    """
    Visualization for cargo instances.
    Reads data from roads and instances directories, and plots interactive
    map showing trips from the instance in 10 sec intervals.
    The map is saved as html in the output folder.


    :param input_path: full path to the directory containing cargo folders
                        'roads' and 'instances'
    :param inst_name:  name of the instance
    :param output_path:
    :return:
    """
    city = inst_name[3:6]
    nodes, edges = read_map(city, path.join(input_path, 'roads'))
    trips = read_instance(path.join(input_path, 'instances', inst_name))
    plot_bokeh_map(trips, nodes, inst_name, output_path)


#TODO instance name as param to make output html file's name
def plot_bokeh_map(trips, nodes, inst_name, output_path):
    """
    Plots interactive map of trips in browser.
    :param df: pandas.DataFrame from cargo instance (id, origin, dest,..., early, late)
    :param nodes: DataFrame/GeoDtatFrame (node id, x, y...)
    :param inst_name:
    :return:
    """
    df = trips[trips.dest >= 0].copy()
    df['dropoff_time'] = df['late']
    df['pickup_time'] = df['early']
    df['minute'] = df['early'] // 10

    nodes_proj = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs=4326)
    nodes_proj = nodes_proj.to_crs(crs=WM)
    # print(nodes_proj.head())

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
    # print(df1.head())

    source1 = ColumnDataSource(data=dict(id=df1.id.values,
                                         pickup_time=df1.early.values, dropoff_time=df1.late.values,
                                         pickup_node=df1.origin.values, dropoff_node=df1.dest.values,
                                         x1=df1.x1.values, y1=df1.y1.values,
                                         x2=df1.x2.values, y2=df1.y2.values,
                                         xx=df1.xx.values, yy=df1.yy.values))
    x_range, y_range = RANGES[inst_name[3:6]]

    plot = figure(plot_width=1200, plot_height=800,
                  x_range=x_range, y_range=y_range,
                  x_axis_type='mercator', y_axis_type='mercator')

    # tiles
    #'CARTODBPOSITRON', 'STAMEN_TERRAIN'
    tile_provider = tile_providers.get_provider('CARTODBPOSITRON_RETINA')
    plot.add_tile(tile_provider)

    # trips
    pickups = plot.circle('x1', 'y1', source=source1, line_alpha=0.9, color='red', size=7)
    dropoffs = plot.circle('x2', 'y2', source=source1, line_alpha=0.9, color='green', size=7)

    routes = plot.multi_line(xs='xx', ys='yy', source=source1, color='darkblue', width=0.8, alpha=0.6,
                             hover_line_alpha=1.0, hover_line_color='blue', hover_line_width=1.2 )

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
    # plot.add_tools(HoverTool(renderers=[pickups],
    #                          tooltips=[('trip id', '@id'),
    #                                    ('pickup time',  '@pickup_time'),
    #                                    ('dropoff time', '@dropoff_time'),
    #                                    ('pickup node',  '@pickup_node')]))
    #
    # plot.add_tools(HoverTool(renderers=[dropoffs],
    #                          tooltips=[('trip id', '@id'),
    #                                    ('pickup time',  '@pickup_time'),
    #                                    ('dropoff time', '@dropoff_time'),
    #                                    ('dropoff node', '@dropoff_node')]))

    plot.add_tools(HoverTool(renderers=[routes],
                             tooltips=[('trip id', '@id'),
                                       ('pickup time',  '@pickup_time'),
                                       ('dropoff time', '@dropoff_time'),
                                       ('pickup node',  '@pickup_node'),
                                       ('dropoff node', '@dropoff_node')]))

    #
    layout = Column(plot, row(time_slider, sizing_mode="stretch_both"))
    page_name = inst_name + '.html'
    output_file(path.join(output_path, page_name),
                          title="Cargo: %s, %s" % (city_name(inst_name), inst_name))
    # add_page_to_index(page_name)
    show(layout)


def city_name(inst_name):
    return CITIES[inst_name[3:6]]


def update_index(map_dir:str, index:str=None):
    """
    Adds all maps from the folder to the table with links in main html file.
    ('index.html' for github pages)
    :param map_dir: folder with generated maps
    :param index: name of the main page with links
    :return:
    """
    index = index if index else 'index.html'
    index_page = path.join(map_dir, index)
    try:
        with open(index_page) as f:
            txt = f.read()
            soup = bs4.BeautifulSoup(txt, features='html.parser')
    except:
        print('Index file %s not found or it is not a valid html.')
        return

    filelnames = listdir(map_dir)
    maps = [m for m in filelnames if 'instance' in m]
    maps.sort()

    original_tag = soup.find(name='ul', id='inst_table')
    original_tag.clear()

    for map in maps:
        new_tag = soup.new_tag('li')
        original_tag.append(new_tag)
        link = soup.new_tag('a', href=map)
        new_tag.append(link)
        link.string = '%s: %s' % (city_name(map), map[:-5])

    with open(index_page, 'w') as f:
        f.write(str(soup))


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


if __name__ == '__main__':

    cargo_path = path.join(getcwd(), 'cargo')

    ny_inst = 'rs-mny-m10k-c3-d6-s10-x1.0.instance' #new york
    bj_inst = 'rs-bj5-m5k-c9-d6-s10-x1.0.instance'  #beijing
    ch_inst = 'rs-cd1-m5k-c3-d6-s10-x1.0.instance'  #chengdu

    show_instance(cargo_path, bj_inst, path.join(getcwd(), 'docs'))
    # show_instance(cargo_path, ny_inst, path.join(getcwd(), 'docs'))
    show_instance(cargo_path, ch_inst, path.join(getcwd(), 'docs'))

    # update index.html
    # update_index('docs')



    # ROADS = path.join(getcwd(), 'cargo', 'roads')
    # CITIES = ['mny', 'bj5', 'cd1']
    # manhattan
    # ny_nodes, ny_edges = read_map(CITIES[0], ROADS)
    # ny_inst = 'rs-mny-m10k-c3-d6-s10-x1.0.instance'
    # ny_trips = read_instance(path.join(getcwd(), 'cargo', 'instances', ny_inst))
    # plot_bokeh_map(ny_trips, ny_nodes)

    #beijing
    # bj_nodes, bj_edges = read_map(CITIES[1], ROADS)
    # bj_inst = 'rs-bj5-m5k-c9-d6-s10-x1.0.instance'
    # bj_trips = read_instance(path.join(getcwd(), 'cargo', 'instances', bj_inst))
    # plot_bokeh_map(bj_trips, bj_nodes)