import folium
from streamlit_folium import folium_static
from sklearn.cluster import DBSCAN
import numpy as np
from geopy.geocoders import Nominatim
import streamlit as st
import requests

def plot_store_locations(df):
    """
    Plot the store locations on a map using latitude and longitude.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None: Displays the map in Streamlit.
    """
    # Create a map centered at the average latitude and longitude
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=6)

    # Add small red circle markers to the map with popup info
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,  # Adjusted size of the circle
            color='black',  # Border color
            fill=True,
            fill_color='red',  # Fill color
            fill_opacity=0.7,
            popup=f"Store: {row['Store']}\nStore Name: {row['Store Name']}\nRegion: {row['Region']}\nPostcode: {row['Store Postcode']}"
        ).add_to(store_map)

    # Display the map in Streamlit
    folium_static(store_map)

def perform_clustering(df, eps_miles):
    """
    Perform DBSCAN clustering on the store locations based on distance.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'Latitude' and 'Longitude' columns.
        eps_miles (float): The maximum distance (in miles) for two points to be considered in the same neighborhood.

    Returns:
        np.ndarray: Cluster labels for each point.
    """
    # Convert miles to kilometers (1 mile = 1.60934 kilometers)
    eps_km = eps_miles * 1.60934

    # DBSCAN requires distance in radians for geographic data
    earth_radius_km = 6371.0
    eps_rad = eps_km / earth_radius_km

    # Convert latitude and longitude to radians
    coords = np.radians(df[['Latitude', 'Longitude']])

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps_rad, min_samples=2, metric='haversine').fit(coords)
    labels = db.labels_

    return labels

def plot_clusters(df, labels):
    """
    Plot clusters on a map using folium.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'Latitude' and 'Longitude' columns.
        labels (np.ndarray): Cluster labels for each point.

    Returns:
        None: Displays the map in Streamlit.
    """
    # Create a map centered at the average latitude and longitude
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=6)

    # Generate color map for clusters
    unique_labels = set(labels)
    colors = [
        f"#{hex(np.random.randint(0, 256))[2:]}{hex(np.random.randint(0, 256))[2:]}{hex(np.random.randint(0, 256))[2:]}"
        for _ in range(len(unique_labels))]

    # Plot each point with a color based on its cluster label
    for idx, row in df.iterrows():
        label = labels[idx]
        color = colors[label] if label != -1 else 'black'  # Black for noise points
        store_ref = row.get('Store', 'Unknown')
        store_name = row.get('Store Name', 'Unknown')
        postcode = row.get('Store Postcode', 'Unknown')
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"Cluster {label}, Store No: {store_ref}, Store Name: {store_name}, Postcode: {postcode}" if label != -1 else f"Noise, Store Ref: {store_ref}, Postcode: {postcode}"
        ).add_to(store_map)

    # Display the map in Streamlit
    folium_static(store_map)

def plot_clusters_by_region(df):
    """
    Plot stores on a map, coloring them by their region.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'Latitude', 'Longitude', and 'Region' columns.

    Returns:
        None: Displays the map in Streamlit.
    """
    # Create a map centered at the average latitude and longitude
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=6)

    # Generate a unique color for each region
    unique_regions = df['Region'].unique()
    region_colors = {region: f"#{np.random.randint(0, 0xFFFFFF):06x}" for region in unique_regions}

    # Plot each store with a color based on its region
    for _, row in df.iterrows():
        region = row['Region']
        region_name = row['Region Name']
        color = region_colors[region]
        store_ref = row.get('Store', 'Unknown')
        store_name = row.get('Store Name', 'Unknown')
        postcode = row.get('Store Postcode', 'Unknown')
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"Region: {region}, Region Name: {region_name}, Store No: {store_ref}, Store Name: {store_name}, Postcode: {postcode}"
        ).add_to(store_map)

    # Display the map in Streamlit
    folium_static(store_map)

def get_coordinates_from_postcode(postcode):
    """
    Converts a postcode to latitude and longitude.

    Parameters:
    - postcode (str): The postcode to convert.

    Returns:
    - (float, float): Latitude and longitude if found, otherwise (None, None).
    """
    geolocator = Nominatim(user_agent="store_locator")
    location = geolocator.geocode(postcode)
    if location:
        return location.latitude, location.longitude
    else:
        st.error("Unable to find coordinates for the given postcode.")
        return None, None

import pandas as pd
from geopy.distance import geodesic

from geopy.distance import geodesic
import pandas as pd


def get_nearby_stores(current_lat, current_lon, stores_df, available_minutes):
    """
    Recommend a list of nearby stores that can be cleaned within the available time slot.
    The function prioritizes stores closest to the given coordinates and calculates distances sequentially
    like a traveling salesman route.

    :param current_lat: Latitude of the starting location (distribution center)
    :param current_lon: Longitude of the starting location
    :param stores_df: DataFrame containing stores information with 'Latitude', 'Longitude', 'Cleaning Time'
    :param available_minutes: Total time available for cleaning stores
    :return: DataFrame with recommended stores and sequential travel distances
    """
    # List to store the visited stores
    route = []

    # Create a copy of the stores DataFrame to avoid modifying the original
    remaining_stores = stores_df.copy()

    # Initialize starting point
    current_location = (current_lat, current_lon)

    while not remaining_stores.empty:
        # Calculate distance from the current location to each remaining store
        remaining_stores["Distance"] = remaining_stores.apply(
            lambda row: geodesic(current_location, (row["Latitude"], row["Longitude"])).miles,
            axis=1
        )

        # Sort by nearest store
        nearest_store = remaining_stores.loc[remaining_stores["Distance"].idxmin()]

        # Check if adding this store exceeds the available cleaning time
        total_cleaning_time = sum([store["Cleaning Time"] for store in route]) + nearest_store["Cleaning Time"]
        if total_cleaning_time > available_minutes:
            break  # Stop adding stores if time exceeds the available slot

        # Add the store to the route
        route.append({
            "Store No": nearest_store["Store"],
            "Store Name": nearest_store["Store Name"],
            "Latitude": nearest_store["Latitude"],
            "Longitude": nearest_store["Longitude"],
            "Distance": nearest_store["Distance"],  # Distance from previous location
            "Cleaning Time": nearest_store["Cleaning Time"],
            "Travel Time": nearest_store["Distance"]*3,
            "Preparation Time": 0.1*nearest_store["Cleaning Time"]
        })

        # Update current location
        current_location = (nearest_store["Latitude"], nearest_store["Longitude"])

        # Remove the selected store from remaining stores
        remaining_stores = remaining_stores.drop(nearest_store.name)

    return pd.DataFrame(route)


# Example usage:
# df_stores = pd.read_csv("your_stores_data.csv")  # Load the dataset
# schedule_df = get_nearby_stores(current_lat, current_lon, df_stores, available_minutes)
# print(schedule_df)


def plot_map_with_routes(current_lat, current_lon, nearby_stores):
    """
    Plots a map with road routes from the current location to each nearby store.

    Parameters:
    - current_lat (float): Latitude of the current location.
    - current_lon (float): Longitude of the current location.
    - nearby_stores (DataFrame): DataFrame of nearby stores with 'Latitude', 'Longitude', and 'Store Id'.

    Returns:
    - folium.Map: Folium map object with markers and routes.
    """
    m = folium.Map(location=[current_lat, current_lon], zoom_start=12)
    folium.Marker(
        [current_lat, current_lon],
        popup="Current Location",
        icon=folium.Icon(color='darkred', icon='shopping-basket', prefix='fa')
    ).add_to(m)

    for _, row in nearby_stores.iterrows():
        store_lat = row['Latitude']
        store_lon = row['Longitude']
        distance = row['Distance']
        store_name = row['Store Name']

        # Fetch the actual road route from OSRM
        route = get_osrm_route(current_lat, current_lon, store_lat, store_lon)

        # Plot the route on the map if found
        if route:
            folium.PolyLine(
                locations=route,
                color="blue",
                weight=2.5,
                opacity=0.8
            ).add_to(m)

        # Determine icon properties based on predicted_resource and Resource_Type
        icon = folium.Icon(color='blue', icon='shopping-basket',
                               prefix='fa')  # Default green shopping basket icon

        # Add a marker for each nearby store
        folium.Marker(
            [store_lat, store_lon],
            popup=f"Store ID: {row['Store Name']}, Distance: {distance:.2f} miles",
            icon=icon
        ).add_to(m)

    return m


import folium
from folium.plugins import AntPath
import pandas as pd

import folium
from folium.plugins import AntPath
import pandas as pd


def plot_store_route(schedule_df, start_lat, start_lon):
    # Extract the first row as the starting (distribution center)
    start_lat, start_lon = schedule_df.iloc[0]["Latitude"], schedule_df.iloc[0]["Longitude"]

    # Create a map centered at the distribution center
    route_map = folium.Map(location=[start_lat, start_lon], zoom_start=10)

    # Add starting location marker with a DISTINCT COLOR (GREEN) and ICON (home)
    folium.Marker(
        location=[start_lat, start_lon],
        popup="🚛 Distribution Center (Start)",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(route_map)

    # Convert DataFrame to list of coordinates [(lat, lon)]
    locations = [(start_lat, start_lon)] + list(zip(schedule_df.iloc[1:]["Latitude"], schedule_df.iloc[1:]["Longitude"]))

    # Plot each store as a marker with BLUE color (skip first row since it's the distribution center)
    for idx, row in schedule_df.iloc[1:].iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"🛒 Stop {idx}: {row['Store Name']} - {row['Cleaning Time']} min",
            icon=folium.Icon(color="blue", icon="shopping-cart"),
        ).add_to(route_map)

    # Draw the optimized route using AntPath (animated route)
    AntPath(locations, color="blue", weight=3, dash_array=[10, 20]).add_to(route_map)

    return route_map



# Example usage:
# Assuming you have `schedule_df` from the `get_nearby_stores` function
# route_map = plot_store_route(schedule_df, current_lat, current_lon)
# route_map.save("cleaning_route_map.html")  # Save as an interactive HTML file


# Example usage:
# Assuming you have `schedule_df` from the `get_nearby_stores` function
# route_map = plot_store_route(schedule_df, current_lat, current_lon)
# route_map.save("cleaning_route_map.html")  # Save as an interactive HTML file


def get_osrm_route(start_lat, start_lon, end_lat, end_lon):
    """
    Get road route between two points using OSRM.

    Parameters:
    - start_lat (float): Latitude of the start location.
    - start_lon (float): Longitude of the start location.
    - end_lat (float): Latitude of the end location.
    - end_lon (float): Longitude of the end location.

    Returns:
    - list of (lat, lon) tuples representing the route.
    """
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    response = requests.get(osrm_url)

    if response.status_code == 200:
        data = response.json()
        route = data['routes'][0]['geometry']['coordinates']
        # OSRM returns coordinates as (lon, lat), so we need to reverse them
        route = [(lat, lon) for lon, lat in route]
        return route
    else:
        st.error("Error fetching route from OSRM.")
        return []
