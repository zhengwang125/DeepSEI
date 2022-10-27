import pickle
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly
from shapely.geometry.point import Point
import shapely
import numpy as np
from scipy.spatial import Voronoi
import geojson
import math

def prepare_points():
    points_df = pd.read_csv('communities_beijing_with_location.csv')
    points_df = points_df.dropna()
    # price = points_df['小区均价']
    x = points_df['lng']
    y = points_df['lat']
    coords = np.vstack((x, y)).T

    vor = Voronoi(coords)
    # fig = voronoi_plot_2d(vor)
    # plt.show()

    lines = [shapely.geometry.LineString(vor.vertices[line]) for line in
             vor.ridge_vertices if -1 not in line]
    polys = shapely.ops.polygonize(lines)
    voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys))
    mjson = geojson.load(open('beijing.geoJson', 'r', encoding='utf-8-sig'))
    geo_df = gpd.GeoDataFrame.from_features(
        mjson["features"]
    )
    # price_sz = pickle.load(open('df_' + tag, 'rb'), encoding='bytes')['price']
    # geo_df['price'] = list(price_sz)  # [1]*len(price_sz)
    # geo_df['price'] = [0 for i in range(len(geo_df))]
    # geo_df = geo_df[['geometry', 'price']]
    voronois = gpd.overlay(voronois, geo_df[['geometry']])
    voronois['price'] = points_df['小区均价']
    voronois = voronois[['geometry', 'price']]
    # voronois.fillna(0,inplace=True)
    pickle.dump(voronois, open('voronois_bj', 'wb'), protocol=2)
    len1 = len(voronois)
    len2 = len(geo_df)
    # voronois = pd.concat([geo_df, voronois],ignore_index=True)
    # voronois = voronois[voronois['price']>0]
    # geo_df = pd.concat([geo_df, voronois],ignore_index=True)

    fig = px.choropleth(voronois,
                        geojson=voronois.geometry,
                        locations= voronois.index,
                        color="price",
                        color_continuous_scale=plotly.colors.colorscale_to_colors([[1.0, 'rgb(255,255,255)'], [1.0, 'rgb(7, 48, 147)']]),
                        # col
                        labels={'price': 'House price'},
                        projection="mercator")
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.update_traces(marker_line_width=0)
    fig.show()

prepare_points()