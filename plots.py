import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import numpy as np


def plot_histogram(data:np.ndarray, num_bins, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.hist(data, bins=num_bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(filename)
    plt.show()


def plot_histograms(data1, data2, numbins, label1, label2, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 12), sharey=True)
    ax = axes[0]
    ax.hist(data1, bins=numbins)
    ax.set_xlabel(label1)
    ax.set_ylabel('Cluster count')
    ax = axes[1]
    ax.hist(data2, bins=numbins)
    ax.set_xlabel(label2)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_trip_counts(counts:pd.DataFrame, col_name: str,  filename: str):
    fig, ax = plt.subplots(1, figsize=(16, 8))
    print(counts.head(20))
    ax.bar(counts.index.values, counts[col_name].values, label='request counts', color='#86bf91')
    for i, v in zip(counts.index.values, counts[col_name].values):
        ax.text(i, 0.9 * v, '%s' % v, fontsize=14, ha='center')

    mean = counts.trip_count.mean()
    max = counts.trip_count.max()
    min = counts.trip_count.min()
    ax.axhline(y=mean, linestyle='dashed', alpha=1, color='#dddddd', zorder=1)
    ax.text(0, mean, 'hourly mean:\n %.2f' % mean, fontsize=16)
    ax.axhline(y=min, linestyle='dashed', alpha=1, color='navy', zorder=1)
    ax.axhline(y=max, linestyle='dashed', alpha=1, color='red', zorder=1)
    ax.set_title('Request counts by hour.')
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.savefig(filename, box_inches='tight')
    plt.show()


def plot_heatmap(x, y, ax):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=512)
    heatmap = gaussian_filter(heatmap, sigma=8)
    jet = cm.get_cmap('jet', 256)
    newcolors = jet(np.linspace(0, 1, 256))
    newcolors[:1, :] = np.zeros(4)
    my_jet = ListedColormap(newcolors)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=my_jet, zorder=2, alpha=0.8)
    ax.scatter(x,y, s=0.2, color='black', zorder=1)


def remove_ticks(ax):
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def draw_graph(G, nodes=True, edges=True, show=False, path=None):

    if not nodes and not edges:
        nodes = True
    print(G.number_of_nodes(), G.number_of_edges())

    fig, ax = plt.subplots()
    if nodes:
        points = [[d['x'], d['y']] for _, d in G.nodes(data=True)]
        ax.scatter([x for x, _ in points], [y for _, y in points], s=5, marker='o')

    if edges:
        lines = [[[G.nodes[u]['x'], G.nodes[u]['y']], [G.nodes[v]['x'], G.nodes[v]['y']]] for u, v, _ in G.edges]
        lc = LineCollection(lines, linewidths=0.5, colors='black')
        ax.add_collection(lc)

    ax.autoscale()
    if path:
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()


def plot_clusters(coords, centroids, colors, filename):

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.scatter(coords.T[0], coords.T[1], c=colors, s=3, zorder=1)
    ax.scatter(centroids.T[0], centroids.T[1], c='gray', s=100, zorder=2)
    for i in range(len(centroids)):
        x, y = centroids[i]
        ax.text(x-0.001, y-0.001, str(i), horizontalalignment='center', size='small', color='black',
             weight='semibold')
    plt.savefig(filename)
    plt.show()


def plot_od_heatmap(trips:gpd.GeoDataFrame, geoname, crs):

    trips = trips.sort_values(by=['time_ms'])
    p_series = gpd.GeoDataFrame(trips, geometry='pn_'+ geoname, crs=crs)
    # print(p_series.head(5))
    d_series = gpd.GeoDataFrame(trips, geometry='dn_' + geoname, crs=crs)
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(28,16))
    plot_heatmap(p_series.geometry.x, p_series.geometry.y, axes[0])
    axes[0].set_title('Pickups')

    plot_heatmap(d_series.geometry.x, d_series.geometry.y, axes[1])
    axes[1].set_title('Dropoffs')
    plt.savefig('heatmap.png')
    plt.show()


    # Density heatmaps hour by hour
    start = 0
    hour_ms = 60*60*1e3 #1 hour
    for r in range(24):
        # for c in range(cols):
        end = start + hour_ms

        df_ = trips[(trips['time_ms'] >= start) & (trips['time_ms'] < end)]
        p_series = gpd.GeoDataFrame(df_, geometry='pn_'+ geoname, crs=crs)
            # print(p_series.head(5))
        d_series = gpd.GeoDataFrame(df_, geometry='dn_'+ geoname, crs=crs)

        psize, dsize  = len(p_series), len(d_series)
        if psize == 0 or dsize == 0:
            break
        fig, axes = plt.subplots(ncols=2, figsize=(24, 16), sharey='all', sharex='all')
        ax = axes[0]
        plot_heatmap(p_series.geometry.x, p_series.geometry.y, ax)
        ax.annotate('Pickup %02d:00-%02d:00' % (r, r+1),xy=(0.1, 1.02), xycoords='axes fraction')
        ax.annotate('%s\ntrips' % psize, xy=(0.85, 0.85), xycoords='axes fraction', fontsize=12)
        ax.tick_params(labelsize=9)
        # remove_ticks(ax)

        ax = axes[1]
        plot_heatmap(d_series.geometry.x, d_series.geometry.y, ax)
        ax.annotate('Dropoff %02d:00-%02d:00' % (r, r+1),xy=(0.1, 1.02), xycoords='axes fraction')
        ax.annotate('%s\ntrips' % psize, xy=(0.85, 0.85), xycoords='axes fraction', fontsize=12)
        ax.tick_params(labelsize=9)
        # remove_ticks(ax)
        start = end
        plt.savefig('demand%s_%s.png' % (r, r+1), bbox_inches='tight')
        plt.show()
