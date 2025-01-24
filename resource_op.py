import folium
from streamlit_folium import folium_static
from sklearn.cluster import DBSCAN
import numpy as np

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
