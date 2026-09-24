import streamlit as st
import pandas as pd
import joblib

model = joblib.load('svr_model.pkl')
scaler = joblib.load('scaler.joblib')

vehicle_class_encoder = {'COMPACT': 0,
 'SUV - SMALL': 11,
 'MID-SIZE': 2,
 'TWO-SEATER': 13,
 'SUBCOMPACT': 10,
 'FULL-SIZE': 1,
 'STATION WAGON - SMALL': 9,
 'SUV - STANDARD': 12,
 'VAN - CARGO': 14,
 'VAN - PASSENGER': 15,
 'PICKUP TRUCK - STANDARD': 6,
 'MINIVAN': 4,
 'SPECIAL PURPOSE VEHICLE': 7,
 'MINICOMPACT': 3,
 'STATION WAGON - MID-SIZE': 8,
 'PICKUP TRUCK - SMALL': 5}

transmission_class_encoder = {'AS5': 13,
 'M6': 24,
 'AV7': 21,
 'AS6': 14,
 'AM6': 7,
 'AM7': 8,
 'AV8': 22,
 'AS8': 16,
 'A7': 4,
 'A6': 3,
 'A8': 5,
 'M7': 25,
 'A4': 1,
 'M5': 23,
 'A5': 2,
 'AV': 18,
 'AS7': 15,
 'A9': 6,
 'AS9': 17,
 'AV6': 20,
 'AS4': 12,
 'AM8': 9,
 'AM9': 10,
 'AS10': 11,
 'A10': 0,
 'AV10': 19}
fuel_type_encoder ={'Z': 4, 'D': 0, 'X': 3, 'E': 1, 'N': 2}

def encode_data(vehicle_class, engine_size, cylinders, transmission, fuel_type, fuel_consumption):
  vehicle_class_label = vehicle_class_encoder[vehicle_class]
  transmission_label = transmission_class_encoder[transmission]
  fuel_type_label = fuel_type_encoder[fuel_type]
  return vehicle_class_label, engine_size, cylinders, transmission_label, fuel_type_label, fuel_consumption

def scale_data(scaler, data):
  return scaler.transform(data) 

st.title("Car CO2 Emission Prediction")

vehicle_class = st.selectbox("Vehicle Class", options=['COMPACT',
 'SUV - SMALL',
 'MID-SIZE',
 'TWO-SEATER',
 'SUBCOMPACT',
 'FULL-SIZE',
 'STATION WAGON - SMALL',
 'SUV - STANDARD',
 'VAN - CARGO',
 'VAN - PASSENGER',
 'PICKUP TRUCK - STANDARD',
 'MINIVAN',
 'SPECIAL PURPOSE VEHICLE',
 'MINICOMPACT',
 'STATION WAGON - MID-SIZE',
 'PICKUP TRUCK - SMALL'])

engine_size = st.number_input("Engine Size (L)", min_value=0.9,max_value=8.4)

cylinders = st.number_input("Cylinders", min_value=3,max_value=16)

transmission = st.selectbox("Transmission",options=['AS5',
 'M6',
 'AV7',
 'AS6',
 'AM6',
 'AM7',
 'AV8',
 'AS8',
 'A7',
 'A6',
 'A8',
 'M7',
 'A4',
 'M5',
 'A5',
 'AV',
 'AS7',
 'A9',
 'AS9',
 'AV6',
 'AS4',
 'AM8',
 'AM9',
 'AS10',
 'A10',
 'AV10'])
fuel_type = st.selectbox("Fuel Type",options=['Z', 'D', 'X', 'E', 'N'])
fuel_consumption = st.number_input("Fuel Consumption (L/100 km)", min_value=2.0)

if st.button("Predict CO2 Emission"):
  encoded_data = encode_data(vehicle_class, engine_size, cylinders, transmission, fuel_type, fuel_consumption)

  data_to_predict = pd.DataFrame([encoded_data], columns=['Vehicle Class','Engine Size(L)', 'Cylinders',  'Transmission', 'Fuel Type','Fuel Consumption Comb (L/100 km)'])

  scaled_data = scale_data(scaler, data_to_predict)

  predicted_co2 = model.predict(scaled_data)[0]

  st.success(f"Predicted CO2 Emission: {predicted_co2:.2f} g/km")
