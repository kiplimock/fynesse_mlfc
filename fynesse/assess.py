from .config import *

from . import access

from .config import *
import osmnx as ox
from osmnx._errors import InsufficientResponseError
import matplotlib.pyplot as plt
import math
import pandas as pd

from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

def plot_city_map(place_name, latitude, longitude, box_size_km=2, poi_tags=None):
    """
    Plot a simple city map with area boundary, buildings, roads, nodes, and optional POIs.

    Parameters
    ----------
    place_name : str
        Name of the place (used for boundary + plot title).
    latitude, longitude : float
        Central coordinates.
    box_size_km : float
        Size of the bounding box in kilometers (default 2 km).
    poi_tags : dict, optional
        Tags dict for POIs (e.g. {"amenity": ["school", "restaurant"]}).
    """

    bbox, nodes, edges, buildings, pois = access.get_osm_datapoints(latitude, longitude, box_size_km, poi_tags)
    west, south, east, north = bbox

    # Area boundary
    area = ox.geocode_to_gdf(place_name).to_crs(epsg=4326)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    area.plot(ax=ax, color="tan", alpha=0.5)
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor="gray", edgecolor="gray", linewidth=0.5)
    edges.plot(ax=ax, color="black", linewidth=1, alpha=0.3, column=None)
    nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3, column=None)
    if pois is not None and not pois.empty:
        pois.plot(ax=ax, color="green", markersize=5, alpha=1, column=None)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_title(place_name, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def get_osm_features(latitude, longitude, box_size_km=2, tags=None):
    """
    Access raw OSM data
    """
    
    # Construct bbox from lat/lon and box_size
    box_height = box_size_km / 111
    box_width = box_size_km / (111 * math.cos(math.radians(latitude)))

    north = latitude + box_height / 2
    south = latitude - box_height / 2
    east = longitude + box_width / 2
    west = longitude - box_width / 2
    bbox = (west, south, east, north)

    # Query OSMnx for features
    if features is None:
        features = []
    feat_keys = set([k for (k, v) in features])
    tags = {key:True for key in feat_keys}

    try:
        pois_df = ox.features_from_bbox(bbox, tags)
    except InsufficientResponseError:
        return {f"{key}_{value or ''}": 0 for key, value in features}

    return pois_df

def get_feature_vector(latitude, longitude, box_size_km=2, features=None):
    """
    Given a central point (latitude, longitude) and a bounding box size,
    query OpenStreetMap via OSMnx and return a feature vector.

    Parameters
    ----------
    latitude : float
        Latitude of the center point.
    longitude : float
        Longitude of the center point.
    box_size : float
        Size of the bounding box in kilometers
    features : list of tuples
        List of (key, value) pairs to count. Example:
        [
            ("amenity", None),
            ("amenity", "school"),
            ("shop", None),
            ("tourism", "hotel"),
        ]

    Returns
    -------
    feature_vector : dict
        Dictionary of feature counts, keyed by (key, value).
    """

    pois_df = get_osm_features(latitude, longitude, box_size_km, features)  

    # Count features matching each (key, value) in poi_types
    feature_vector = {}
    for key, value in features:
        if key in pois_df.columns:
            if value is None:
                feature_vector[f"{key}_"] = pois_df[key].notnull().sum()
            else:
                feature_vector[f"{key}_{value}"] = (pois_df[key] == value).sum()
        else:
            feature_vector[f"{key}_{value or ''}"] = 0

    # Return dictionary of counts
    return feature_vector

    raise NotImplementedError("Feature extraction not implemented yet.")

def build_feature_dataframe(city_dicts, features, box_size_km=1):
    results = {}
    for country, cities in city_dicts:
        for city, coords in cities.items():
            vec = get_feature_vector(
                coords["latitude"],
                coords["longitude"],
                box_size_km=box_size_km,
                features=features
            )
            vec["country"] = country
            results[city] = vec
    return pd.DataFrame(results).T

def visualize_feature_space(X, y, method='PCA'):
    """
    Assess data distribution and separability
    """
    if method == 'PCA':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_reduced = reducer.fit_transform(X)

    plt.figure(figsize=(8,6))
    for country, color in [("Kenya", "green"), ("England", "blue")]:
        mask = (y == country)
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1],
                    label=country, color=color, s=100, alpha=0.7)
    
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D projection of feature vectors")
    plt.legend()
    plt.show()


def print_err(target, pred_dict):
    """
    Prints error metrics between target and predictions.
    Args:
        target (dict): Dictionary containing true values with keys 'train', 'val', 'test'.
                       Each key maps to another dict with keys 'X', 'Y', 'A', and 'B'.
        pred_dict (dict): Dictionary containing predicted values with keys 'train', 'val', 'test'.
                          Each key maps to a tuple of (predicted_magnitudes, predicted_angles).
    
    Returns:
        err_dict (dict): Dictionary containing error metrics for each dataset split.
    """
    err_dict = {}
    for key in pred_dict.keys():
        cmplx_target = target[key]['Y'][:,:,0]*np.exp(1j*target[key]['Y'][:,:,1])
        if pred_dict[key][0].shape == target[key]['Y'][:,:,0].shape:
            mag_err = pred_dict[key][0] - target[key]['Y'][:,:,0]
            ang_err = pred_dict[key][1] - target[key]['Y'][:,:,1]
            cmplx_pred = pred_dict[key][0]*np.exp(1j*pred_dict[key][1])
        else:
            print("shape mismatch so broadcasting")
            mag_err = pred_dict[key][0][0,:] - target[key]['Y'][:,:,0]
            ang_err = pred_dict[key][1][0,:] - target[key]['Y'][:,:,1]
            cmplx_pred = pred_dict[key][0][0,:]*np.exp(1j*pred_dict[key][1][0,:])

        cmplx_err = cmplx_target - cmplx_pred
        print(key)
        print('Vm MAE: %.5e' % (np.abs(mag_err).mean()))
        print('Vm MSE: %.5e' % (np.square(mag_err).mean()))
        print('Vm FVU: %.5e' % (np.square(mag_err).mean()/target[key]['Y'][:,:,0].var()))
        print("")

        print('Va MAE: %.5e' % (np.abs(ang_err).mean()))
        print('Va MSE: %.5e' % (np.square(ang_err).mean()))
        print('Va FVU: %.5e' % (np.square(ang_err).mean()/target[key]['Y'][:,:,1].var()))
        print("")
        print('Mean Complex Err. Mag.: %.5e \n' % (np.abs(cmplx_err).mean()))
        tmp = {
            'Vm MAE':np.abs(mag_err).mean(),
            'Vm MSE':np.square(mag_err).mean(),
            'Vm FVU':np.square(mag_err).mean()/target[key]['Y'][:,:,0].var(),
            'Va MAE':np.abs(ang_err).mean(),
            'Va MSE':np.square(ang_err).mean(),
            'Va FVU':np.square(ang_err).mean()/target[key]['Y'][:,:,1].var(),
            'MCEM':np.abs(cmplx_err).mean(),
        }
        err_dict.update({key:tmp})
    
    return err_dict


def plot_comparison_bar(errors, err_key, methods, sel_key=None, *, show_mag=True, show_ang=True, logy=True):
    rows = len(errors[0].keys()) if sel_key is None else len(sel_key)
    cols = int(show_mag)+int(show_ang)
    if err_key == 'MCEM':
        cols = 1
        show_ang = False
        show_mag = False
    if cols == 0:
        raise ValueError("No columns in figure")
    fig, axs = plt.subplots(rows, cols, sharex=True)
    bar_colors = [
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown'
        'tab:pink',
        'tab:gray',
        'tab:olive'
        'tab:cyan',
        ]*5
    
    
    sel_key = list(errors[0].keys()) if sel_key is None else sel_key
    if rows == 1 and cols == 1:
        sel_col = 'Vm' if show_mag else 'Va'
        col_key = sel_col+' '+err_key
        if err_key == 'MCEM':
            sel_col = 'Mean Complex Err Mag.'
            col_key = err_key
        counts = [error[sel_key[0]][col_key] for error in errors]
        if logy:
            axs.set_yscale('log')
        axs.bar(
                    methods,
                    counts,
                    #label=methods, 
                    color=bar_colors[:len(counts)]
                )
        axs.set_ylabel(err_key)
        axs.set_title(sel_col)
        print(counts)
        #axs.legend(title='Methods')
    elif rows > 1:
        for i, j in enumerate(sel_key):
            if show_mag and show_ang:
                counts = [error[j]['Vm '+err_key] for error in errors]
                if logy:
                    axs[i,0].set_yscale('log')
                    axs[i,1].set_yscale('log')
                axs[i,0].bar(
                    methods,
                    counts,
                    #label=methods, 
                    color=bar_colors[:len(counts)]
                )
                axs[i,0].set_ylabel(err_key)
                if i == 0:
                    axs[i,0].set_title('Vm')
                    #axs[i,0].legend(title='Methods')

                counts = [error[j]['Va '+err_key] for error in errors]
                axs[i,1].bar(
                    methods,
                    counts,
                    #label=methods, 
                    color=bar_colors[:len(counts)]
                )
                #axs[i,1].set_ylabel(err_key)
                if i == 0:
                    axs[i,1].set_title('Va')
                    #axs[i,1].legend(title='Methods')
            else:
                sel_col = 'Vm' if show_mag else 'Va'
                col_key = sel_col+' '+err_key
                if err_key == 'MCEM':
                    sel_col = 'Mean Complex Err Mag.'
                    col_key = err_key
                counts = [error[sel_key[0]][col_key] for error in errors]
                if logy:
                    axs[i].set_yscale('log')
                axs[i].bar(
                    methods,
                    counts,
                    #label=methods, 
                    color=bar_colors[:len(counts)]
                )
                axs[i].set_ylabel(err_key)
                if i == 0:
                    axs[i].set_title(sel_col)
                    #axs[i].legend(title='Methods')

    plt.show()