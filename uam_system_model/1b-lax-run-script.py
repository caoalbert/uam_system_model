import numpy as np
from gurobipy import *
import sys
import os
import pickle
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
from geopy.distance import geodesic

from uam_system_model.StarNetwork import StarNetwork

from uam_system_model.Pricing import PricingOptimizer

import argparse

# for debug mode
# os.chdir(os.getcwd()+"/economic-feasibility")

# pkill -f "run_pricing.py"
# nohup python run_pricing.py > run_pricing.out &


def calculate_distances_in_meters(coords):
    num_points = len(coords)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = geodesic(coords[i], coords[j]).meters / 1600
    return distance_matrix

def ensure_directory_exists(path):
    """Ensure the directory exists; if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_name", "-n", type=str, default="pricing_vot_substitution_with_repo_test")
    parser.add_argument("--output_folder_path", "-o", type=str, default='pricing_output/atrds/')
    parser.add_argument("--optimality_gap", "-g", type=float, default=0.05)
    parser.add_argument("--fleet_size_range", "-f", type=str, default="10-10")
    args = parser.parse_args()
    ensure_directory_exists(args.output_folder_path)


    coordinates = np.array(
        [
            (33.94417072663332, -118.40248449470265),
            (34.043119432990345, -118.26720629762752),
            (33.81606030845284, -118.15122129260827),
            (34.17160155814583, -118.60525790222589),
            (34.13843197335217, -118.35511311047983),
            (33.812102161009804, -117.91897753216998),
            (34.09174681098575, -118.32725521957076),
            (34.08807589928915, -118.55063573741867),
            (34.06666969690935, -118.4114352069469),
        ]
    )

    distance_matrix = calculate_distances_in_meters(coordinates)
    vertiports = ["LAX", "DTLA", "LGB", "WDHL", "UVS", "ANH", "HWD", "PSD", "BVH"]

    vertiport_pmf = np.array([0, 6.1, 3.3, 1.5, 1.5, 5.4, 4.3, 2.5, 2.2])

    # vertiport_pmf = np.array([0, 3.8, 2.4, 1, 1, 4.1, 2.5, 2.2, 1.5])

    vertiport_pmf = vertiport_pmf / vertiport_pmf.sum()

    flight_time_matrix = np.load("../UAM-System-Model/uam_system_model/data/config/flight_time_matrix.npy")
    energy_consumption_matrix = np.load(
        "../UAM-System-Model/uam_system_model/data/config/energy_consumption_matrix.npy"
    )

    network = StarNetwork(
        vertiport_names=vertiports,
        flight_distance_matrix=distance_matrix,
        flight_time_matrix=flight_time_matrix ,
        energy_consumption_matrix=energy_consumption_matrix,
    )

    SURVEY_P_TNC = 0.33327  # from the recorded data

    total_seat = network.demand_generator.yearly_capacity[0] + network.demand_generator.yearly_capacity[1]
    OD_PAX_TO_SEAT_RATIO = 77500000 / total_seat # from the recorded data

    VERTIPORT_PMF = np.array([0, 3.8, 2.4, 1, 1, 4.1, 2.5, 2.2, 1.5])/ 100 # from passenger origin survey data

    vertiport_density = np.zeros((len(VERTIPORT_PMF), len(VERTIPORT_PMF)))
    vertiport_density[0, :] = VERTIPORT_PMF
    vertiport_density[:, 0] = VERTIPORT_PMF

    p_uam_array = 1 * SURVEY_P_TNC * OD_PAX_TO_SEAT_RATIO
    p_uam_array = p_uam_array * (np.repeat(vertiport_density[:, :, np.newaxis], 24, axis=2))

    p_uam_array = np.concatenate([[p_uam_array[0,:,:] , p_uam_array[:,0,:]]])
    network.load_demand(
        month=9,
        day=5,
        vertiport_pmf=p_uam_array,
        max_waiting_time=10,
        auto_regressive_alpha=0.7,
        seed=9,
    )


    op = PricingOptimizer(
        StarNetwork=network
    )

    with open("config/travel_time.pkl", "rb") as file:
        uber_travel_time, first_mile, last_mile, distance_avg = pickle.load(file)
    
    first_or_last_distance = np.zeros(shape=(9,9,24))
    first_or_last_distance[0,:,:] = distance_avg
    first_or_last_distance[:,0,:] = distance_avg
    uam_flight_time = np.load("../UAM-System-Model/uam_system_model/data/config/flight_time_matrix.npy") * 5


    fleet_size_range = args.fleet_size_range.split("-")
    for fleet_size in range(int(fleet_size_range[0]), int(fleet_size_range[1])+1, 10):
        print(f"Running for fleet size: {fleet_size}")

        df = op.optimize(time_resolution=30, 
                        num_vehicles=fleet_size, 
                        uber_travel_time=uber_travel_time, 
                        first_mile_time=first_mile, 
                        last_mile_time=last_mile, 
                        first_or_last_distance=first_or_last_distance,
                        uam_flight_time=uam_flight_time, 
                        uam_distance_matrix=distance_matrix,
                        optimality_gap=args.optimality_gap,
                        value_of_time = 32.63,
                        utility_type="betas",
                        uber_fare=np.array([0, 50, 62, 78, 57, 77, 57, 57, 45]),
                        time_limit=3600*4
                        )
    
        df.to_csv(os.path.join(args.output_folder_path, args.output_name+f"_{fleet_size}.csv"), index=False)