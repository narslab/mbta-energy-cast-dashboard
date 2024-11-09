import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import ast
import os
import scipy.stats as stats
import tensorflow as tf

# Load the model and data
model = load_model('models/test_model.h5')
estimate_dist = pd.read_csv("data/tidy/estimated_distributions_planning_metrics.csv")
df_test = pd.read_csv("data/tidy/test_data.csv")

# Set the wide representation as the deafult
st.set_page_config(layout="wide")
# Set up Streamlit
st.title("MBTA EnergyCast Dashboard")
# st.write("")
st.markdown("<p style='font-size: 18px; font-weight: bold;'>A energy forecasting tool for the Massachusetts Bay Transportation Authority urban rail transit system (Boston T)</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 25px; font-weight: bold;'>About</p>", unsafe_allow_html=True)
st.markdown(
    """
    <p style="font-size: 18px; margin-top: 20px;">
    This planning tool allows you to specify desired planning metrics for the MBTA URT for each line. 
    When you click on the "Predict" button, it runs a long short-term memory network (LSTM) model to 
    generate forecasts for the next 90 days. The current model has been trained on data from 01-2019 to 09-2021.
    </p>
    """,
    unsafe_allow_html=True
)
# Mitigate the button
st.markdown(
    """
    <style>
    .stButton > button {
        font-size: 30px;
        padding: 20px 40px;
        color: red;
        border-radius: 15px; 
        float: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Create two columns
col1, col2 = st.columns(2)

with col1:
    # Left side: User-defined metric values
    st.header("Daily Planning Metrics")
    
    st.markdown("""
    <style>
    .reduced-margin {
        margin-bottom: -10px; /* Adjust this value to control the spacing */
    }
    </style>
    """, unsafe_allow_html=True)

    # Define the lines and their default values
    lines = ["Red", "Blue", "Orange", "Green", "Mattapan"]
    default_values = {
        "num_trips": [5, 3, 4, 6, 7],
        "distance": [10, 12, 8, 9, 11],
        "speed": [2, 1, 3, 2.5, 1.5]
    }

    # Display headers for the metrics
    col_label, col_trips, col_distance, col_speed = st.columns([1, 2, 2, 2])
    col_label.markdown("<p style='font-size: 18px; font-weight: bold;'>Line</p>", unsafe_allow_html=True)
    col_trips.markdown("<p style='font-size: 18px; font-weight: bold;'>Number of Trips</p>", unsafe_allow_html=True)
    col_distance.markdown("<p style='font-size: 18px; font-weight: bold;'>Operating Distance (miles)</p>", unsafe_allow_html=True)
    col_speed.markdown("<p style='font-size: 18px; font-weight: bold;'>Average Speed (mph)</p>", unsafe_allow_html=True)

    # Initialize a dictionary to store the metrics
    metrics = {}

    # Row inputs for each line
    for i, line in enumerate(lines):
        # Create a row with four columns for each line
        col_label, col_trips, col_distance, col_speed = st.columns([1, 2, 2, 2])
        
        # Display the line name in the first column
        col_label.markdown(f"<p style='font-size: 15px; font-weight: bold;'>{line}</p>", unsafe_allow_html=True)
        
        # Display the input fields in the remaining columns
        num_trips = col_trips.number_input('', key=f"{line}_num_trips", value=default_values["num_trips"][i])
        distance = col_distance.number_input('', key=f"{line}_distance", value=default_values["distance"][i])
        speed = col_speed.number_input('', key=f"{line}_speed", value=default_values["speed"][i])
        
        # Store the values in the metrics dictionary
        metrics[line] = {'Num_trips': num_trips, 'Dist': distance, 'Avg_speed': speed}

    # Prediction button
    st.button("Run Model")

    # Text before the logos
    st.markdown(
    """
    <p style="font-size: 16px; margin-top: 20px;">
    This tool was developed by the <a href="https://narslab.org/" target="_blank">Networks for Accessibility, Resilience and Sustainability Laboratory (NARS Lab)</a> at 
    the University of Massachusetts Amherst, in partnership with the <a href="https://www.mbta.com/" target="_blank">Massachusetts Bay Transportation Authority (MBTA)</a>. 
    The project was funded by the <a href="https://www.mass.gov/orgs/massachusetts-department-of-transportation" target="_blank">Massachusetts Department of Transportation (MassDOT)</a>, 
    and the grant was administered by the <a href="https://www.umasstransportationcenter.org/umtc/default.asp" target="_blank">University of Massachusetts Transportation Center (UMTC)</a>.
    </p>
    
    <p style="font-size: 16px; font-weight: bold; margin-top: 20px;">Credits:</p>
    <ul style="font-size: 15px;">
        <li>Lead Researcher and Developer: Zhuo Han (NARS Lab, UMass)</li>
        <li>Co-Principal Investigators: Dr. Eleni Christofa and Dr. Eric Gonzales (UMass)</li>
        <li>Principal Investigator: <a href="https://people.umass.edu/jboke/" target="_blank">Dr. Jimi Oke</a> (NARS Lab, UMass)</li>
        <li>Project Champion: Sean Donaghy (MBTA)</li>
        <li>Project Manager: Michael Flanary (MassDOT)</li>
        <li>Administrative Support: Kimberley Foster, Matt Mann, Michelle Clark (UMTC) and Anil S. Gurcan (MassDOT)</li>
    </ul>

    <p style="font-size: 15px; font-weight: bold;">
        <a href="https://github.com/narslab/mbta-energy-cast-dashboard.git" target="_blank">GitHub Repository: MBTA EnergyCast Dashboard</a>
    </p>
    """, 
    unsafe_allow_html=True
)

    # Display images from the 'logo' folder below the inputs
    logo_path = "logo"  # Path to the 'logo' folder
    image_width = 200  # Set a fixed width for all images

    # Specify the image filenames in the desired order
    ordered_images = ["TREEM.png", "NARSLAB.png", "UMTC.png", "MBTA.png", "MassDOT.png"]

    # Create a row of columns, one for each image, and display them in the specified order
    # Split the ordered images into two rows
    row1_images = ordered_images[:3]  # First three images for the first row
    row2_images = ordered_images[3:]  # Remaining images for the second row

    # Display the first row of images
    cols1 = st.columns(len(row1_images))
    for col, image_file in zip(cols1, row1_images):
        image_path = os.path.join(logo_path, image_file)
        if os.path.exists(image_path):
            # Apply smaller width for NARSLAB
            width = 150 if image_file == "NARSLAB.png" else image_width
            col.image(image_path, width=width)
    # Display the second row of images
    cols2 = st.columns(len(row2_images))
    for col, image_file in zip(cols2, row2_images):
        image_path = os.path.join(logo_path, image_file)
        if os.path.exists(image_path):
            col.image(image_path, width=image_width)

with col2:
    # Data generation process
    estimate_dist['parameters'] = estimate_dist['parameters'].apply(ast.literal_eval)
    def generate_data(dist_name, params, size=90):
        dist = getattr(stats, dist_name)
        return dist.rvs(size=size, **params)
    generated_data = {'route_id': [], 'metric': [], 'generated_values': []}
    for _, row in estimate_dist.iterrows():
        route_id = row['route_id']
        metric = row['metric']
        dist_name = row['distribution']
        params = row['parameters']
        data = generate_data(dist_name, params)
        generated_data['route_id'].append(route_id)
        generated_data['metric'].append(metric)
        generated_data['generated_values'].append(data)
    generated_data_df = pd.DataFrame(generated_data)

    # Update planning variables based on generated data
    planning_var = pd.DataFrame(metrics).T.reset_index()
    planning_var.columns = ['Line', 'Num_trips', 'Dist', 'Avg_speed']
    planning_var = pd.concat([planning_var] * 90, ignore_index=True)

    metric_to_column = {'trip_diff': 'Num_trips', 'dist_diff': 'Dist', 'avg_speed_diff': 'Avg_speed'}
    for _, row in generated_data_df.iterrows():
        line = row['route_id']
        metric = row['metric']
        generated_values = row['generated_values']
        column = metric_to_column[metric]
        line_mask = planning_var['Line'] == line
        planning_var.loc[line_mask, column] += generated_values[:line_mask.sum()]

    planning_var_reshaped = planning_var.pivot_table(index=planning_var.index // 5, columns='Line', values=['Num_trips', 'Dist', 'Avg_speed']).reset_index(drop=True)
    planning_var_reshaped.columns = [f"{metric}_{line}" for metric, line in planning_var_reshaped.columns]

    # Combine data for model input
    combined_data = pd.concat([planning_var_reshaped.reset_index(drop=True), df_test.reset_index(drop=True)], axis=1)

    # Preprocess data for model input
    seq_data = combined_data[['Energy', 'TAVG']]
    non_seq_data = combined_data.drop(['Energy', 'TAVG'], axis=1)
    non_seq_data = non_seq_data.loc[:, non_seq_data.columns != 'Unnamed: 0']

    seq_scaler = StandardScaler()
    non_seq_scaler = StandardScaler()
    seq_data_scaled = seq_scaler.fit_transform(seq_data)
    non_seq_data_scaled = non_seq_scaler.fit_transform(non_seq_data)

    # Convert sequential data into sequences of 10 time steps
    time_steps = 1
    seq_input = np.array([seq_data_scaled[i:i + time_steps] for i in range(len(seq_data_scaled) - time_steps + 1)])
    non_seq_input = non_seq_data_scaled[time_steps - 1:]

    # Make predictions
    predictions = model.predict([seq_input, non_seq_input])
    predictions_original_scale = seq_scaler.inverse_transform(predictions)
    energy_values = predictions_original_scale[:, 0]
    temp_values = predictions_original_scale[:, 1]

    # First row with the energy and temperature forecast plots
    st.markdown("<h2 style='text-align: center;'>Forecast</h2>", unsafe_allow_html=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot for energy forecast
    ax1.plot(energy_values, marker='o', linestyle='-', color='b')
    ax1.set_ylabel("Energy (MWh)", fontsize=15)
    ax1.set_xlabel("Day", fontsize=15)
    ax1.grid(True)
    
    # Plot for temperature forecast
    ax2.plot(temp_values, marker='o', linestyle='-', color='r')
    ax2.set_ylabel("Temperature (F)", fontsize=15)
    ax2.set_xlabel("Day", fontsize=15)
    ax2.grid(True)
    
    # Display the combined figure
    st.pyplot(fig)
    # The overall energy
    overall_energy_consumption = np.sum(energy_values)

    # Display the total energy consumption in a single line
    st.markdown(
        f"<p style='font-size: 20px; font-weight: bold;'>Overall Energy Consumption (MWh): "
        f"<span style='font-weight: normal;'>{overall_energy_consumption:.2f}</span></p>",
        unsafe_allow_html=True
    )
    # Right side: Statistics table for daily energy based on predictions
    st.header("Daily Energy Forecast Statistics")
    daily_energy_stats = {
        'Statistic': ['Average', 'Median', 'Min', 'Max'],
        'Energy (MWh)': [
            round(np.mean(energy_values), 2),
            round(np.median(energy_values), 2),
            round(np.min(energy_values), 2),
            round(np.max(energy_values), 2)
        ]
    }
    energy_stats_df = pd.DataFrame(daily_energy_stats)
    energy_stats_df['Energy (MWh)'] = energy_stats_df['Energy (MWh)'].map("{:.2f}".format)

    st.table(energy_stats_df)


