import pandas as pd
import streamlit as st
import plotly.express as px
import resource_op as ro
import folium
from streamlit_folium import st_folium
import requests
import random
from streamlit_folium import folium_static

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def calculate_cleaning_capacity(df, window_hours, cleaning_rates):
    """
    Calculate how many stores can be cleaned within a given cleaning window.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing store data.
        window_hours (int): Total cleaning hours in the window.
        cleaning_rates (dict): Time required (in minutes) to clean trolleys and baskets.

    Returns:
        pd.DataFrame: A DataFrame showing cleaning schedules for the stores.
    """
    # Convert window hours to minutes
    total_minutes = window_hours * 60

    # Calculate cleaning time for each store
    df['Cleaning Time (mins)'] = (
        (df['Total Trolleys'] / 15) * cleaning_rates['trolley'] +
        (df['Rollable Basket'] / 15) * cleaning_rates['rollable_basket'] +
        (df['Handheld Basket'] / 15) * cleaning_rates['handheld_basket']
    )

    # Determine the number of stores that can be cleaned within the time limit
    df['Cumulative Time'] = df['Cleaning Time (mins)'].cumsum()
    df['Within Window'] = df['Cumulative Time'] <= total_minutes

    # Filter stores that can be cleaned within the window
    cleaned_stores = df[df['Within Window']]
    return cleaned_stores

def calculate_people_needed(df, window_hours, cleaning_rates, break_time=0):
    """
    Calculate the number of people needed to clean all stores within the given cleaning window.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing store data.
        window_hours (int): Total cleaning hours in the window.
        cleaning_rates (dict): Time required (in minutes) to clean trolleys and baskets.
        break_time (int): Break time in minutes to be subtracted from the available time.

    Returns:
        int: Total number of people needed.
    """
    # Total cleaning time for all stores in minutes
    total_cleaning_time = (
        (df['Total Trolleys'].sum() / 15) * cleaning_rates['trolley'] +
        (df['Rollable Basket'].sum() / 15) * cleaning_rates['rollable_basket'] +
        (df['Handheld Basket'].sum() / 15) * cleaning_rates['handheld_basket']
    )

    # Available time per person in minutes (subtract break time)
    available_time_per_person = (window_hours * 60) - break_time

    # Calculate the number of people needed
    people_needed = total_cleaning_time / available_time_per_person
    return int(people_needed) + (1 if people_needed % 1 != 0 else 0)


# OSRM route function
def get_osrm_route(start_lat, start_lon, end_lat, end_lon):
    """
    Get road route between two points using OSRM.
    """
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    response = requests.get(osrm_url)

    if response.status_code == 200:
        data = response.json()
        route = data['routes'][0]['geometry']['coordinates']
        # Reverse coordinates from (lon, lat) to (lat, lon)
        route = [(lat, lon) for lon, lat in route]
        return route
    else:
        st.error("Error fetching route from OSRM.")
        return []

# Plot the map with the route
def plot_route_on_map(stores):
    """
    Plot filtered stores and the route between them on a map using Folium.
    """
    if len(stores) < 2:
        st.warning("At least two stores are required to calculate a route.")
        return None

    # Create a Folium map centered at the first store
    start_lat, start_lon = stores.iloc[0]['Latitude'], stores.iloc[0]['Longitude']
    route_map = folium.Map(location=[start_lat, start_lon], zoom_start=10)

    # Add markers for each store
    for _, row in stores.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"Store: {row['Store']}<br>Store Name: {row['Store Name']}<br>Postcode: {row['Store Postcode']}",
            icon=folium.Icon(color='blue', icon='fa-shopping-basket', prefix='fa'),
        ).add_to(route_map)

    # Calculate the route between consecutive stores
    for i in range(len(stores) - 1):
        route = get_osrm_route(
            stores.iloc[i]['Latitude'], stores.iloc[i]['Longitude'],
            stores.iloc[i + 1]['Latitude'], stores.iloc[i + 1]['Longitude']
        )
        folium.PolyLine(route, color="red", weight=2.5, opacity=1).add_to(route_map)

    return route_map

def create_schedule(stores, store_names, store_postcodes, cleaning_times, slot_minutes, days=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], break_time=0):
    """
    Create a cleaning schedule for stores with varying cleaning times.

    :param stores: List of store IDs/names.
    :param cleaning_times: List of cleaning times corresponding to each store (in minutes).
    :param slot_minutes: Total available minutes per slot.
    :param days: Days of the week available for scheduling.
    :param break_time: Break time (minutes) to account for during the slot.
    :return: Pandas DataFrame with the schedule.
    """
    # Initialize variables
    schedule = []
    current_day = 0
    remaining_minutes = slot_minutes - break_time

    # Iterate through stores and their cleaning times
    for store, store_name, store_postcode, cleaning_time in zip(stores, store_names, store_postcodes, cleaning_times):
        # Check if the current store can fit in the remaining time for the day
        if remaining_minutes >= cleaning_time:
            schedule.append({'Store': store,'Store Name': store_name,'Store Postcodes': store_postcode, 'Estimated Cleaning Time': cleaning_time, 'Day': days[current_day], 'Slot': f'{slot_minutes // 60} hours'})
            remaining_minutes -= cleaning_time
        else:
            # Move to the next day
            current_day = (current_day + 1) % len(days)
            remaining_minutes = slot_minutes - break_time - cleaning_time
            schedule.append({'Store': store, 'Store Name': store_name,'Store Postcodes': store_postcode, 'Estimated Cleaning Time': cleaning_time, 'Day': days[current_day], 'Slot': f'{slot_minutes // 60} hours'})

    return pd.DataFrame(schedule)

def create_schedule_year(stores, cleaning_times, frequencies, slot_minutes, days=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], break_time=0):
    """
    Create a cleaning schedule for stores with varying cleaning times and cleaning frequencies.

    :param stores: List of store IDs/names.
    :param cleaning_times: List of cleaning times corresponding to each store (in minutes).
    :param frequencies: List of cleaning frequencies (e.g., bi-monthly, monthly, quarterly).
    :param slot_minutes: Total available minutes per slot.
    :param days: Days of the week available for scheduling.
    :param break_time: Break time (minutes) to account for during the slot.
    :return: Pandas DataFrame with the schedule.
    """
    # Frequency to occurrences per year and corresponding months
    frequency_mapping = {
        'Monthly': range(1, 13),  # Every month
        'Bi-monthly': range(1, 13, 2),  # Every two months
        'Quarterly': range(1, 13, 3)  # Every three months
    }

    # Initialize schedule
    schedule = []
    current_day_idx = 0
    remaining_minutes = slot_minutes - break_time

    # Iterate through each month
    for month in range(1, 13):
        for store, cleaning_time, frequency in zip(stores, cleaning_times, frequencies):
            # Check if the store needs cleaning in this month
            if month in frequency_mapping[frequency]:
                # Find the next available day for this store
                while remaining_minutes < cleaning_time:
                    # Move to the next day and reset time
                    current_day_idx = (current_day_idx + 1) % len(days)
                    remaining_minutes = slot_minutes - break_time

                # Schedule the store
                schedule.append({
                    'Store': store,
                    'Month': month,
                    'Day': days[current_day_idx],
                    'Slot': f'{slot_minutes // 60} hours',
                    'Estimated Cleaning Time (minutes)': cleaning_time,
                    'Frequency': frequency
                })
                # Deduct time from the current day's slot
                remaining_minutes -= cleaning_time

    return pd.DataFrame(schedule)

if __name__ == "__main__":

    df = load_data("updated_lidl.csv")

    # Convert Store column to string
    df["Store"] = df["Store"].astype(str)
    st.title("Lidl Resource Optimisation")
    st.dataframe(df)
    st.subheader("Number of Stores by Region")

    region_counts = df["Region"].value_counts().reset_index()
    region_counts.columns = ["Region", "Number of Stores"]
    fig_bar = px.bar(
        region_counts,
        x="Region",
        y="Number of Stores",
        color="Region",
        title="Number of Stores by Region",
        text="Number of Stores",
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar)

    st.subheader("Treemap of Stores by Region")
    fig_treemap = px.treemap(
        region_counts,
        path=["Region"],
        values="Number of Stores",
        title="Treemap of Stores by Region",
    )
    st.plotly_chart(fig_treemap)

    st.subheader("Total Trolleys and Baskets by Region")
    fig = px.bar(
        df,
        x="Region",
        y=["Total Trolleys", "Total Baskets"],
        title="Comparison of Trolleys and Baskets by Region",
        labels={"value": "Count", "variable": "Type"},
        barmode="group",
        text_auto=True
    )
    st.plotly_chart(fig)

    # Selection for Pie Chart
    display_option = st.selectbox(
        "Select Proportion to Display:",
        ["Proportion of Stores with Trolleys", "Proportion of Stores with Baskets"]
    )

    if display_option == "Proportion of Stores with Trolleys":
        pie_trolleys = px.pie(
            df,
            names="Trolleys (Y/N)",
            title="Proportion of Stores with Trolleys",
            hole=0.3
        )
        st.plotly_chart(pie_trolleys)
    else:
        pie_baskets = px.pie(
            df,
            names="Baskets (Y/N)",
            title="Proportion of Stores with Baskets",
            hole=0.3
        )
        st.plotly_chart(pie_baskets)

    # Initialize session state for map and results
    if "map" not in st.session_state:
        st.session_state.map = None
    if "results" not in st.session_state:
        st.session_state.results = None

    if st.checkbox("Plot clusters by Region"):
        # Get clustering parameters

        # # Perform clustering
        # labels = ro.perform_clustering(df, eps_miles)
        # df['Cluster'] = labels  # Add cluster labels to the DataFrame
        #
        # # Plot clusters
        # st.write(f"Clusters based on a {eps_miles} mile threshold:")
        ro.plot_clusters_by_region(df)

    # Streamlit UI
    st.subheader("Region Selector with Route Mapping")

    # Create a dropdown for regions
    regions = df['Region Name'].unique()
    selected_region = st.selectbox("Select a region:", options=regions)

    # Filter stores based on the selected region
    filtered_stores = df[df['Region Name'] == selected_region]

    # Display the selected region and filtered stores
    st.write(f"Selected Region: {selected_region}")
    st.write("Stores in this region:")
    st.dataframe(filtered_stores)

    # Cleaning rates in minutes
    cleaning_rates = {
        'trolley': 15,
        'rollable_basket': 15,
        'handheld_basket': 10
    }

    # Calculate and display for 5 am to 9 am (4 hours)
    st.write("Cleaning Capacity for 5 am to 9 am")
    cleaned_stores_morning = calculate_cleaning_capacity(filtered_stores, window_hours=4, cleaning_rates=cleaning_rates)
    st.write("Stores cleaned in the morning:", cleaned_stores_morning)

    # Calculate and display for 5 am to 9 pm (16 hours)
    st.write("Cleaning Capacity for 5 am to 9 pm")
    cleaned_stores_day = calculate_cleaning_capacity(filtered_stores, window_hours=16, cleaning_rates=cleaning_rates)
    st.write("Stores cleaned in the day:", cleaned_stores_day)

    # Calculate people needed
    st.write("Number of People needed to clean the entire Region")
    people_needed_morning = calculate_people_needed(filtered_stores, window_hours=4, cleaning_rates=cleaning_rates)
    st.write(f"People needed for 5 am to 9 am: {people_needed_morning}")

    people_needed_day = calculate_people_needed(filtered_stores, window_hours=16, cleaning_rates=cleaning_rates,
                                                break_time=30)
    st.write(f"People needed for 5 am to 9 pm (including 30-minute break): {people_needed_day}")

    # Button to generate the route
    if st.button("Optimize Route"):
        if filtered_stores.empty:
            st.warning("No stores are available in the selected region.")
            st.session_state.map = None
        else:
            st.session_state.results = filtered_stores
            st.session_state.map = plot_route_on_map(filtered_stores)

    # Display the map from session state
    if st.session_state.map:
        st.subheader("Shortest Route to visit all stores in the region")
        st_folium(st.session_state.map, width=700, height=500)

    if not filtered_stores.empty:
        # Convert the filtered data into the required format for scheduling
        stores = filtered_stores["Store"]
        store_names = filtered_stores["Store Name"]
        store_postcodes = filtered_stores["Store Postcode"]
        cleaning_times = filtered_stores["Cleaning Time"]
        frequencies = filtered_stores["Frequency of Basket Clean"]

        # Generate schedule for 5 am to 9 am slot (240 minutes)
        schedule_5am_9am = create_schedule(stores, store_names, store_postcodes, cleaning_times, slot_minutes=240)

        # Generate schedule for 5 am to 9 pm slot (960 minutes with a 30-minute break)
        schedule_5am_9pm = create_schedule(stores, store_names, store_postcodes, cleaning_times, slot_minutes=960, break_time=30)

        # Display the schedules
        st.subheader(f"Schedule for 5 am to 9 am slot for {selected_region}:")
        st.dataframe(schedule_5am_9am)

        st.subheader(f"Schedule for 5 am to 9 pm slot for {selected_region}:")
        st.dataframe(schedule_5am_9pm)
    else:
        st.warning("No stores are available in the selected region.")

    if not filtered_stores.empty:
        stores = filtered_stores['Store']
        cleaning_times = [random.randint(10, 50) for _ in range(len(stores))]
        frequencies = [random.choice(['Monthly', 'Bi-monthly', 'Quarterly']) for _ in range(len(stores))]
        schedule = create_schedule_year(stores, cleaning_times, frequencies, slot_minutes=240)
        schedule1 = create_schedule_year(stores, cleaning_times, frequencies, slot_minutes=960)
        st.subheader("Yearly Schedule 5 am to 9 am slot")
        st.dataframe(schedule)
        st.subheader("Yearly Schedule 5 am to 9 pm slot")
        st.dataframe(schedule1)

        postcode = st.text_input("Enter Start Postcode:", "BR1 1EZ")
        time_slot = st.radio("Select Cleaning Slot:", ["5 AM - 9 AM", "5 AM - 9 PM"])

        # Define available time based on user selection
        available_minutes = 240 if time_slot == "5 AM - 9 AM" else 960
        current_lat, current_lon = ro.get_coordinates_from_postcode(postcode)

        # Get stores that can be cleaned within available time
        nearby_stores = ro.get_nearby_stores(current_lat, current_lon, filtered_stores, available_minutes)

        # Print or return the stores
        st.write(nearby_stores)

        if not nearby_stores.empty:
            st.write("Map with Routes to Nearby Stores:")
            m = ro.plot_map_with_routes(current_lat, current_lon, nearby_stores)
            folium_static(m)
