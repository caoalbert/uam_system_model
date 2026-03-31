import numpy as np
from gurobipy import *
import sys
import os
import pickle
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from geopy.distance import geodesic
from metadata.uam_schema import UAMSchema

from uam_system_model.StarNetworkJFK import StarNetwork

from uam_system_model.Pricing import PricingOptimizer

import argparse

import pandas as pd
import geopandas as gpd

import time

# pkill -f "run_pricing_ram.py"
# nohup python run_pricing_ram.py > run_pricing.out &

SCHEMA = UAMSchema()

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


    vertiports = ["UIU", "CHI", "ORD", "MDW"]

    flight_distance_matrix = np.array([
        [0, 129, 150, 100],
        [129, 0, 30, 40],
        [150, 30, 0, 50],
        [100, 40, 50, 0]
    ])  # in miles
    flight_time_matrix = np.array([
        [0, 70, 80, 55],
        [70, 0, 15, 20],
        [80, 15, 0, 25],
        [55, 20, 25, 0]
    ])  # in minutes
    energy_consumption_matrix = np.array([
        [0, 0.5, 0.5, 0.5],
        [0.5, 0, 0.5, 0.5],
        [0.5, 0.5, 0, 0.5],
        [0.5, 0.5, 0.5, 0]
    ])  # in kWh

    network = StarNetwork(
        vertiport_names=vertiports,
        flight_distance_matrix=flight_distance_matrix,
        flight_time_matrix=flight_time_matrix,
        energy_consumption_matrix=energy_consumption_matrix,
    )

    def make_od(csv_path, origin, destination):
        df = pd.read_csv(csv_path)
        df['hour'] = pd.to_datetime(df[SCHEMA.START_TIME]).dt.hour
        od_df = df.groupby('hour').size().reset_index(name='total_trips')
        od_df['od'] = f"{origin}_{destination}"
        return od_df
    
    df_uiu_chi = make_od("data/uiuc_c.csv", "UIU", "CHI")
    df_chi_uiu = make_od("data/c_uiuc.csv", "CHI", "UIU")
    df_uiu_ord = make_od("data/uiuc_ord.csv", "UIU", "ORD")
    df_ord_uiu = make_od("data/ord_uiuc.csv", "ORD", "UIU")
    df_uiu_mdw = make_od("data/uiuc_mdw.csv", "UIU", "MDW")
    df_mdw_uiu = make_od("data/mdw_uiuc.csv", "MDW", "UIU")

    df = pd.concat([df_uiu_chi, df_chi_uiu, df_uiu_ord, df_ord_uiu, df_uiu_mdw, df_mdw_uiu], ignore_index=True)

    df['total_trips'] *= 365

    network.load_demand(df)

    driving_travel_time = np.zeros(shape=(len(vertiports), len(vertiports), 24))
    driving_cost = np.zeros(shape=(len(vertiports), len(vertiports), 24))

    first_mile = np.zeros(shape=(len(vertiports), 24)) 
    last_mile = np.zeros(shape=(len(vertiports), 24))
    first_or_last_cost = np.zeros(shape=(len(vertiports), len(vertiports), 24))  # Assuming first and last mile costs are included in the driving cost

    def fill_od(csv_path, o, d):
        tmp = pd.read_csv(csv_path)
        tmp['hour'] = pd.to_datetime(tmp[SCHEMA.START_TIME]).dt.hour
        od_df = tmp.groupby('hour').agg({"hour": "size", "Driving_IVTT_min": "mean", "FM_duration_min": "mean", "LM_duration_min": "mean", "Driving_Fare_USD": "mean", "FM_fare_USD": "mean", "LM_fare_USD": "mean"})
        od_df = od_df.rename(columns={"hour": "total_trips"})
        od_df = od_df.reset_index()
        for _, row in od_df.iterrows():
            h = int(row['hour'])
            driving_travel_time[o,d,h] = row["Driving_IVTT_min"]
            first_mile[o,h] = row["FM_duration_min"]
            last_mile[d,h] = row["LM_duration_min"]
            driving_cost[o,d,h] = row["Driving_Fare_USD"]
            first_or_last_cost[o,d,h] = row["FM_fare_USD"] + row["LM_fare_USD"]

    fill_od("data/uiuc_c.csv", 0, 1)
    fill_od("data/c_uiuc.csv", 1, 0)
    fill_od("data/uiuc_ord.csv", 0, 2)
    fill_od("data/ord_uiuc.csv", 2, 0)
    fill_od("data/uiuc_mdw.csv", 0, 3)
    fill_od("data/mdw_uiuc.csv", 3, 0)

    op = PricingOptimizer(StarNetwork=network)

    fs = np.arange(10, 65, 5)
    opex_per_asm = np.arange(0.6, 1.6, 0.1)

    for f in fs:
        for c in opex_per_asm:
            print(f"Running optimization for fleet size {f} and CASM {c}...")
            df = op.optimize(
            time_resolution=30,
            num_vehicles=f,
            uber_travel_time=driving_travel_time,
            uber_fare=driving_cost,
            first_mile_time=first_mile,
            last_mile_time=last_mile,
            first_or_last_cost=first_or_last_cost,  # Assuming first and last mile costs are included in the driving cost
            uam_flight_time = flight_time_matrix,
            uam_distance_matrix=flight_distance_matrix,
            optimality_gap=0.05,
            value_of_time=32.63,
            time_limit=3600,
            uam_transition_time=10,
            utility_type="betas",
            opex_per_asm=c,
            fato_capacity = 10,  # capacity per time interval for FATO constraint
            num_seats=4,
            fixed_cost_per_flight=20,
            verbose=False
            )
            df[0].to_csv(f"results2/{f}_{round(c*10, 0)}.csv", index=False)
            df[1].to_csv(f"results2/task_log_{f}_{round(c*10, 0)}.csv", index=False)