from .config import *
import osmnx as ox
import matplotlib.pyplot as plt
import math

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

    # Convert km to degrees
    lat_offset = (box_size_km / 2) / 111
    lon_offset = (box_size_km / 2) / (111 * math.cos(math.radians(latitude)))

    north = latitude + lat_offset
    south = latitude - lat_offset
    east = longitude + lon_offset
    west = longitude - lon_offset
    bbox = (west, south, east, north)

    # Area boundary
    area = ox.geocode_to_gdf(place_name).to_crs(epsg=4326)

    # Road graph
    graph = ox.graph_from_bbox(bbox, network_type="all")
    nodes, edges = ox.graph_to_gdfs(graph)

    # Buildings & POIs
    buildings = ox.features_from_bbox(bbox, tags={"building": True})
    pois = None
    if poi_tags:
        pois = ox.features_from_bbox(bbox, tags=poi_tags)

    # Ensure correct geometry column
    nodes = nodes.set_geometry("geometry")
    edges = edges.set_geometry("geometry")
    buildings = buildings.set_geometry("geometry")
    if pois is not None:
        pois = pois.set_geometry("geometry")

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
