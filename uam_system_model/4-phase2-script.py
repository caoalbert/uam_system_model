import os
import argparse
import numpy as np
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from uam_system_model.StarNetwork import StarNetwork
from uam_system_model.FleetOpVRP import FleetOpVRP
import pickle

# pkill -f "run_network_eco_with_pricing.py"
# nohup python run_network_eco_with_pricing.py > run_network_eco_with_pricing.out &

def ensure_directory_exists(path):
    """Ensure the directory exists; if not, create it."""
    if not os.path.exists(path):
        os.makedirs(path)

def optimize(vrp, fleet_size, mzpt, mte, beta_0, output_folder_path, n_processes):
    unserved_tours, served_tours = vrp.optimize(
        fleet_size, 
        mzpt, 
        mte*100, 
        beta_0_coefficient=beta_0,
        n_processes=n_processes, 
        epsilon=0,
        verbose=False,
    )
    # vrp.demand_post_clustering.to_csv(f"{output_folder_path}/{fleet_size}_{int(mzpt)}_{int(mte*10)}_post_clustering.csv", index=False)
    served_tours.to_csv(f"{output_folder_path}/{fleet_size}_{int(mzpt)}_{int(mte*10)}_{beta_0}.csv", index=False)

def calculate_distances_in_meters(coords):
    num_points = len(coords)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = geodesic(coords[i], coords[j]).meters / 1600
    return distance_matrix
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", "-s", type=bool, default=True)
    parser.add_argument("--n_processes", "-n", type=int, default=36)
    # original network_class_routing_input_scale_1_26.pkl
    parser.add_argument("--network_input_file", "-i", type=str, default="atrds/pricing_70.pkl")
    parser.add_argument("--output_folder_path", "-o", type=str, default='output/atrds/with_pricing_70/')
    parser.add_argument("--fleet_size_range", "-f", type=str, default='70-70')
    # max_zero_prep_time is the maximum time allowed for zero charging time in specified in the clustering step
    parser.add_argument("--max_zero_prep_time_range", "-mzp", type=str, default='6-6')
    # max_tour_energy is the maximum energy allowed for a tour
    parser.add_argument("--max_tour_energy_range", "-mte", type=str, default='0.4-0.4')
    # beta 0 is the coefficient for the calculating the prep time
    parser.add_argument("--beta_0", "-b0", type=str, default='4-4')
    args = parser.parse_args()
    ensure_directory_exists(args.output_folder_path)

    with open(f"config/{args.network_input_file}", "rb") as file:
        network = pickle.load(file)

    vrp = FleetOpVRP(StarNetwork=network)

    file_list = os.listdir(args.output_folder_path)
    all_files = []
    file_names = np.empty(shape=(0, 4))
    for i in file_list:
        i = i.split('.csv')[0]
        fleet_size = i.split('_')[0]
        mzpt = i.split('_')[1]
        mte = i.split('_')[2]
        beta_0 = i.split('_')[3]
        if beta_0 != 'post':
            file_names = np.vstack((file_names, np.array([fleet_size, mzpt, mte, beta_0])))
    file_names = file_names.astype(int)
    input_to_function = []

    fleet_size_range = args.fleet_size_range.split('-')
    fleet_size_range = np.arange(int(fleet_size_range[0]), int(fleet_size_range[1]) + 1, 1)
    mzpt_range = args.max_zero_prep_time_range.split('-')
    mzpt_range = np.arange(int(mzpt_range[0]), int(mzpt_range[1]) + 1, 1)
    mte_range = args.max_tour_energy_range.split('-')
    mte_range = np.arange(float(mte_range[0]), float(mte_range[1]) + 0.1, 0.1)
    beta_0_range = args.beta_0.split('-')
    beta_0_range = np.arange(int(beta_0_range[0]), int(beta_0_range[1]) + 1, 1)

    for fleet_size in fleet_size_range:
        for mzpt in mzpt_range:
            for mte in mte_range:
                for beta_0 in beta_0_range:
                    if (np.all(file_names == np.array([fleet_size,mzpt,10*mte,beta_0]), axis=1).any() == False):
                        input_to_function.append((fleet_size, mzpt, mte, beta_0))

    print(f"Expecting {len(input_to_function)} runs")
    for fleet_size, mzpt, mte, beta_0 in input_to_function:
        optimize(vrp, fleet_size, mzpt, mte, beta_0, args.output_folder_path, args.n_processes)


    


