pip install tensorflow == "2.11.0"
import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import h5py
from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf
print(tf.__version__)
# Load the weights into the modified model
model= load_model('../models/test_model.h5')


st.title("Sustainable decision-making tool for MBTA URT system (SDTMR)")

# Sequence Input (Energy and Temperature)
st.header("Sequential inputs:")
energy = st.number_input("Initial Energy", value=0.0)
temperature = st.number_input("Initial Temperature", value=0.0)

# Create a placeholder for the initial sequence of 10 timesteps (all values initially set to the input values)
seq_data = np.array([[energy, temperature]] * 10).reshape(1, 10, 2)

# Non-sequence Input (Trips, Speed, Distance, Ridership)
st.header("Non-sequential inputs:")

# Input for number of trips by Line
num_trips_red = st.number_input("Number of Trips (Red Line)", value=0)
num_trips_blue = st.number_input("Number of Trips (Blue Line)", value=0)
num_trips_orange = st.number_input("Number of Trips (Orange Line)", value=0)
num_trips_green = st.number_input("Number of Trips (Green Line)", value=0)
num_trips_mattapan = st.number_input("Number of Trips (Mattapan Line)", value=0)

# Input for average speed by Line
avg_speed_red = st.number_input("Average Speed (Red Line)", value=0.0)
avg_speed_blue = st.number_input("Average Speed (Blue Line)", value=0.0)
avg_speed_orange = st.number_input("Average Speed (Orange Line)", value=0.0)
avg_speed_green = st.number_input("Average Speed (Green Line)", value=0.0)
avg_speed_mattapan = st.number_input("Average Speed (Mattapan Line)", value=0.0)

# Input for operating distance by Line
op_dist_red = st.number_input("Operating Distance (Red Line)", value=0.0)
op_dist_blue = st.number_input("Operating Distance (Blue Line)", value=0.0)
op_dist_orange = st.number_input("Operating Distance (Orange Line)", value=0.0)
op_dist_green = st.number_input("Operating Distance (Green Line)", value=0.0)
op_dist_mattapan = st.number_input("Operating Distance (Mattapan Line)", value=0.0)

ridership = st.number_input("Ridership", value=0)

# Prepare the non-sequence input data (shape: (1, 16))
non_seq_data = np.array([[num_trips_red, num_trips_blue, num_trips_orange, num_trips_green, num_trips_mattapan,
                          avg_speed_red, avg_speed_blue, avg_speed_orange, avg_speed_green, avg_speed_mattapan,
                          op_dist_red, op_dist_blue, op_dist_orange, op_dist_green, op_dist_mattapan, ridership]])



# Button to make predictions
if st.button("Predict"):
    # Initialize StandardScaler for sequence and non-sequence data
    seq_scaler = StandardScaler()
    non_seq_scaler = StandardScaler()

    # Reshape the sequence data to 2D (10, 2) and scale
    seq_data_reshaped = seq_data.reshape(-1, 2)  # Reshape (1, 10, 2) -> (10, 2)
    seq_data_scaled_reshaped = seq_scaler.fit_transform(seq_data_reshaped)  # Fit and transform
    seq_data_scaled = seq_data_scaled_reshaped.reshape(1, 10, 2)  # Reshape back to (1, 10, 2)

    # Scale the non-sequence data (2D already)
    non_seq_data_scaled = non_seq_scaler.fit_transform(non_seq_data)


    n_steps = 90  # Number of future steps to predict
    predictions = []

    for i in range(n_steps):
        # Predict the next step using the scaled data
        next_prediction_scaled = model.predict([seq_data_scaled, non_seq_data_scaled])

        # Inverse transform the predictions to original scale
        next_prediction_original = seq_scaler.inverse_transform(next_prediction_scaled)

        # Store the original-scale prediction
        predictions.append(next_prediction_original)

        # Shift the sequence and add the predicted scaled value as the new last timestep
        seq_data_scaled = np.roll(seq_data_scaled, -1, axis=1)  # Shift sequence to the left
        seq_data_scaled[0, -1, :] = next_prediction_scaled  # Insert the scaled prediction into the sequence

    # Convert predictions list to a numpy array for easier handling
    predictions = np.array(predictions)

    # Separate energy and temperature from predictions
    predicted_energy = predictions[:, 0, 0]  # Assuming energy is the first output
    predicted_temperature = predictions[:, 0, 1]  # Assuming temperature is the second output

    # Create a time series (for the predicted future timesteps)
    timesteps = np.arange(1, n_steps + 1)

    # Plot predictions for energy
    fig_energy, ax_energy = plt.subplots()
    ax_energy.plot(timesteps, predicted_energy, label='Predicted Energy', color='b')
    ax_energy.set_xlabel("Timesteps")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("Predicted Energy over Time")
    ax_energy.legend()

    # Plot predictions for temperature
    fig_temp, ax_temp = plt.subplots()
    ax_temp.plot(timesteps, predicted_temperature, label='Predicted Temperature', color='r')
    ax_temp.set_xlabel("Timesteps")
    ax_temp.set_ylabel("Temperature")
    ax_temp.set_title("Predicted Temperature over Time")
    ax_temp.legend()

    # Display the plots
    st.pyplot(fig_energy)
    st.pyplot(fig_temp)
