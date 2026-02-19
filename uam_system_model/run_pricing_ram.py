import numpy as np
from gurobipy import *
import sys
import os
import pickle
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from geopy.distance import geodesic

from uam_system_model.StarNetworkJFK import StarNetwork

from uam_system_model.Pricing import PricingOptimizer

import argparse

import pandas as pd
import geopandas as gpd

import time

# pkill -f "run_pricing_ram.py"
# nohup python run_pricing_ram.py > run_pricing.out &

def ensure_directory_exists(path):
    """Ensure the directory exists; if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder_path", "-o", type=str, default='results2/')
    parser.add_argument("--fleet_size_range", "-f", type=str, default='30-60')
    parser.add_argument("--casm_range", "-c", type=str, default='0.6-1.0')
    parser.add_argument("--optimality_gap", "-g", type=float, default=0.05)
    parser.add_argument("--time_limit", "-t", type=int, default=3600*2)
    args = parser.parse_args()
    ensure_directory_exists(args.output_folder_path)


    vertiports = ["UIU", "CHI"]

    flight_distance_matrix = np.array([[0, 129],
                                    [129, 0]])  # in miles
    flight_time_matrix = np.array([[0, 70],
                                [70, 0]])  # in minutes
    energy_consumption_matrix = np.array([[0, 0.5],
                                        [0.5, 0]])  # in kWh

    network = StarNetwork(
        vertiport_names=vertiports,
        flight_distance_matrix=flight_distance_matrix,
        flight_time_matrix=flight_time_matrix,
        energy_consumption_matrix=energy_consumption_matrix,
    )

    df = pd.read_csv("data/c_uiuc.csv")
    df['hour'] = pd.to_datetime(df['trip_start_time']).dt.hour
    c_uiuc = df.groupby('hour').size().reset_index(name='total_trips')
    c_uiuc['od'] = 'UIU_CHI'
    df = pd.read_csv("data/uiuc_c.csv")
    df['hour'] = pd.to_datetime(df['trip_start_time']).dt.hour
    uiuc_c = df.groupby('hour').size().reset_index(name='total_trips')
    uiuc_c['od'] = 'CHI_UIU'
    df = pd.concat([c_uiuc, uiuc_c], ignore_index=True)

    df['total_trips'] *= 365

    network.load_demand(df)

    uber_travel_time = np.zeros(shape=(len(vertiports), len(vertiports), 24))
    uber_fare = np.zeros(shape=(len(vertiports), len(vertiports), 24))
    first_or_last_distance = np.zeros(shape=(len(vertiports), len(vertiports), 24))

    first_mile = np.zeros(shape=(len(vertiports), 24)) 
    last_mile = np.zeros(shape=(len(vertiports), 24))

    df = pd.read_csv("data/uiuc_c.csv")
    df['hour'] = pd.to_datetime(df['trip_start_time']).dt.hour
    c_uiuc = df.groupby('hour').size().reset_index(name='total_trips')

    df_grouped = df.groupby("hour").agg({"hour": "size", "Driving_IVTT_min": "mean", "FM_duration_min": "mean", "LM_duration_min": "mean", "Driving_Fare_USD": "mean", "FM_fare_USD": "mean", "LM_fare_USD": "mean"})
    df_grouped = df_grouped.rename(columns={"hour": "total_trips"})
    df_grouped = df_grouped.reset_index()


    for _, row in df_grouped.iterrows():
        hour = int(row['hour'])
        uber_travel_time[0, 1, hour] = row['Driving_IVTT_min']
        first_mile[0, hour] = row['FM_duration_min']
        last_mile[1, hour] = row['LM_duration_min']
        uber_fare[0, 1, hour] = row['Driving_Fare_USD']

    df = pd.read_csv("data/c_uiuc.csv")
    df['hour'] = pd.to_datetime(df['trip_start_time']).dt.hour
    c_uiuc = df.groupby('hour').size().reset_index(name='total_trips')

    df_grouped = df.groupby("hour").agg({"hour": "size", "Driving_IVTT_min": "mean", "FM_duration_min": "mean", "LM_duration_min": "mean", "Driving_Fare_USD": "mean", "FM_fare_USD": "mean", "LM_fare_USD": "mean"})
    df_grouped = df_grouped.rename(columns={"hour": "total_trips"})
    df_grouped = df_grouped.reset_index()

    for _, row in df_grouped.iterrows():
        hour = int(row['hour'])
        uber_travel_time[1, 0, hour] = row['Driving_IVTT_min']
        first_mile[1, hour] = row['FM_duration_min']
        last_mile[0, hour] = row['LM_duration_min']
        uber_fare[1, 0, hour] = row['Driving_Fare_USD']

    first_or_last_distance = np.array([[0, 4], [8, 0]])  # in miles
    first_or_last_distance = np.repeat(first_or_last_distance[:, :, np.newaxis], 24, axis=2)


    op = PricingOptimizer(StarNetwork=network)

    fs = np.arange(10, 65, 5)
    casm = np.arange(0.6, 0.9, 0.1)

    for f in fs:
        for c in casm:
            print(f"Running optimization for fleet size {f} and CASM {c}...")
            df = op.optimize(
                time_resolution=30,
                num_vehicles=f,
                uber_travel_time=uber_travel_time,
                uber_fare=uber_fare,
                first_mile_time=first_mile,
            last_mile_time=last_mile,
            first_or_last_distance=first_or_last_distance,
            uam_flight_time = flight_time_matrix,
            uam_distance_matrix=flight_distance_matrix,
            optimality_gap=0.05,
            value_of_time=32.63,
            time_limit=3600,
            utility_type="vot",
            CASM=c,
            verbose=False
        )
            df.to_csv(f"results2/{f}_{round(c*10, 0)}.csv", index=False)