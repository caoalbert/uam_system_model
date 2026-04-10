import logging
import math
import multiprocessing
import re
from time import time

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB, Model
from tqdm import tqdm


class FleetOpVRP:
    def __init__(
        self,
        StarNetwork,
    ):
        self.network = StarNetwork
        self.flight_time = self.network.flight_time
        self.energy_consumption = self.network.energy_consumption

        schedule = StarNetwork.schedule.copy()
        schedule["schedule"] = schedule["schedule"] // 5
        schedule["origin"] = schedule["od"].apply(
            lambda x: self.network.vertiport_dict[x.split("_")[0]]
        )
        schedule["destination"] = schedule["od"].apply(
            lambda x: self.network.vertiport_dict[x.split("_")[1]]
        )

        schedule = (
            schedule.groupby(["schedule", "origin", "destination"])
            .agg(
                count=("origin", "count"),
                revenue=("revenue", "sum"),
                num_pax=("num_pax", "sum"),
            )
            .reset_index()
            .rename(columns={"schedule": "time"})
        )

        schedule["per_flight_revenue"] = schedule["revenue"] / schedule["count"]
        schedule["per_flight_num_pax"] = schedule["num_pax"] / schedule["count"]
        self.flight_demand = schedule

    def optimize(
        self,
        num_vehicles,
        max_zero_prep_time,
        max_tour_energy,
        beta_0_coefficient,
        epsilon=0,
        linear_battery_charging=False,
        n_processes=16,
        verbose=False,
        alns_iterations=30,
    ):
        """
        param num_vehicles: Number of vehicles in the fleet
        param max_zero_prep_time: Maximum flight duration under which no preparation time is required (in 5 minute intervals)
        param max_tour_energy: Maximum energy consumption for a tour (in fraction of battery capacity)
        param epsilon: Maximum time difference between flights in a subtour (in 5 minute intervals)
        param linear_battery_charging: If True, use linear battery charging model
        param n_processes: Number of parallel processes to use for solving the VRP
        param verbose: If True, print verbose output during network construction
        param alns_iterations: Number of ALNS iterations for post-processing (0 disables ALNS, falls back to single-pass insertion)
        """

        self.linear = linear_battery_charging
        self.num_vehicles = num_vehicles
        self.max_zero_prep_time = max_zero_prep_time
        self.max_tour_energy = max_tour_energy
        self.beta_0_coefficient = beta_0_coefficient
        self.epsilon = epsilon
        # verbose network construction, 1-vehicle problem
        self.verbose = verbose

        run_id = f"M{self.network.month}_D{self.network.day}_V{num_vehicles}_MZPT{max_zero_prep_time}_MTE{int(max_tour_energy)}_B0{beta_0_coefficient}"
        logger = logging.getLogger(f"FleetOpVRP_{run_id}")
        logger.setLevel(logging.INFO)
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        start_time = time()
        logger.info("[1] BUILDING SUBTOURS")
        list_subtours = self._BuildSubtours()

        logger.info(f"[2] SOLVING KNAPSACK PROBLEM")
        selected_lists = self._BinPacking(list_subtours)
        demand2 = self._UpdateDemand(self.flight_demand, selected_lists, list_subtours)
        #
        logger.info(
            f"NUMBER OF FLIGHTS BEFORE: {self.flight_demand['count'].sum()} NUMBER OF TOURS AFTER: {demand2['count'].sum()}"
        )
        self.demand2 = demand2.copy()

        logger.info(f"[3] SOLVING THE CLUSTERING/SCHEDULING PROBLEM")
        demand2 = self._SolveClusteringProblem(demand2)
        num_flights_selected = demand2[demand2["cluster_id"] > -1]["tour_length"].sum()
        logger.info(
            f"NUMBER OF FLIGHTS SELECTED AFTER CLUSTERING: {int(num_flights_selected)}/{int(self.flight_demand['count'].sum())}"
        )
        self.demand_post_clustering = demand2.copy()
        demand2[["duration", "time"]] = demand2[["duration", "time"]] * 5

        logger.info("[4] SOLVING 1-VRP")
        unserved_tours, served_tours = self._run_parallel_vrp(demand2, n_processes)
        self.unserved_tours = unserved_tours
        self.served_tours = served_tours
        logger.info(
            f"NUMBER OF SERVED FLIGHTS: {int(served_tours['tour_length'].sum())} / {int(num_flights_selected)}"
        )

        if alns_iterations > 0 and not unserved_tours.empty:
            logger.info(f"[5] ALNS POST-PROCESSING ({alns_iterations} iterations)")
            unserved_tours, served_tours = self._alns(
                unserved_tours, served_tours, n_iter=alns_iterations
            )
        else:
            logger.info("[5] Iterative Insertion (single-pass)")
            unserved_tours, served_tours = self._crossover(unserved_tours, served_tours)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(
            f"TOTAL NUMBER OF SERVED FLIGHTS: {int(served_tours['tour_length'].sum())}. SERVED RATIO: {served_tours['tour_length'].sum() / self.flight_demand['count'].sum()}"
        )
        end_time = time()
        logger.info(f"Total time taken: {(end_time - start_time)/60} minutes \n")

        return unserved_tours, served_tours

    def solve_exact(self, num_vehicles, time_limit=3600, optimality_gap=0.01):
        """
        Solve the full eUAMVRP-NL to (near-)optimality using Gurobi as a MIP benchmark.
        Intended for small instances only (≤20 flight tasks, ≤3 vehicles).

        Must be called after optimize() has run (so self.demand2 is populated), or
        after manually calling _BuildSubtours, _BinPacking, and _UpdateDemand.

        Returns
        -------
        obj_val   : float  - best objective value found
        mip_gap   : float  - final MIP gap
        runtime_s : float  - wall-clock solve time in seconds
        served    : DataFrame
        unserved  : DataFrame
        """
        if not hasattr(self, "demand2"):
            raise RuntimeError(
                "Call optimize() first (or build demand2 manually) before solve_exact()."
            )
        demand = self.demand2.copy()
        demand[["duration", "time"]] = demand[["duration", "time"]] * 5

        m, not_served = self._RunVRP(
            demand,
            K=num_vehicles,
            resolve=False,
            time_limit=time_limit,
            optimality_gap=optimality_gap,
        )

        idx_served = [i for i in range(len(demand)) if i not in not_served]
        served = demand.iloc[idx_served].copy().reset_index(drop=True)
        unserved = demand.iloc[not_served].copy().reset_index(drop=True)

        return m.ObjVal, m.MIPGap, m.Runtime, served, unserved

    def _BuildSubtours(self):
        # max_tour_energy (in fraction of battery capacity)
        # flight_time (in 5 minute intervals)
        # energy_consumption (in fraction of battery capacity

        list_subtours = []
        subtour = []

        for idx_init, row_init in self.flight_demand.iterrows():
            energy_consumed = self.energy_consumption[
                int(row_init["origin"]), int(row_init["destination"])
            ]
            time_ready = (
                row_init["time"]
                + self.flight_time[
                    int(row_init["origin"]), int(row_init["destination"])
                ]
            )
            total_time = self.flight_time[
                int(row_init["origin"]), int(row_init["destination"])
            ]
            prev_dest = row_init["destination"]
            subtour = [idx_init]

            for idx, row in self.flight_demand.iterrows():
                if idx <= 0:
                    continue

                time = row["time"]
                origin = row["origin"]
                # Subtour conditions
                if (
                    ((time - time_ready) >= 0 and (time - time_ready) <= self.epsilon)
                    and energy_consumed
                    + self.energy_consumption[
                        int(row["origin"]), int(row["destination"])
                    ]
                    < self.max_tour_energy
                    and prev_dest == origin
                ):
                    energy_consumed += self.energy_consumption[
                        int(row["origin"]), int(row["destination"])
                    ]
                    time_ready = (
                        time
                        + self.flight_time[int(row["origin"]), int(row["destination"])]
                    )
                    total_time += self.flight_time[
                        int(row["origin"]), int(row["destination"])
                    ]
                    prev_dest = row["destination"]
                    subtour.append(idx)

            if len(subtour) > 1:
                list_subtours.append(subtour)

        return list_subtours

    def _BinPacking(self, list_subtours):
        occurrence_limits = {}
        for idx, row in self.flight_demand.iterrows():
            occurrence_limits[idx] = row["count"]

        model = gp.Model()
        vars = []
        for i in range(len(list_subtours)):
            vars.append(model.addVar(vtype=GRB.BINARY, name=f"list_{i}"))
        # Objective function: maximize the number of selected items/trips
        model.setObjective(
            gp.quicksum(
                vars[i] * len(list_subtours[i]) for i in range(len(list_subtours))
            ),
            GRB.MAXIMIZE,
        )

        # Constraint: the number of tours selected is within the demand limit
        for number, limit in occurrence_limits.items():
            model.addConstr(
                gp.quicksum(
                    vars[i] * list_subtours[i].count(number)
                    for i in range(len(list_subtours))
                )
                <= limit,
                f"limit_{number}",
            )

        model.setParam("OutputFlag", 0)
        model.optimize()
        selected_lists = [i for i in range(len(list_subtours)) if vars[i].x > 0.5]
        return selected_lists

    def _UpdateDemand(self, demand, selected_lists, list_subtours):
        demand = demand.copy()
        # for subtour in selected_lists:
        combined_nodes = pd.DataFrame(
            columns=[
                "time",
                "origin",
                "destination",
                "count",
                "duration",
                "energy_consumption",
                "num_pax",
                "tour_revenue",
                "tour_length",
                "tour_sequence",
            ]
        )
        for idx in selected_lists:
            subtour = list_subtours[idx]
            tour_sequence = ""
            time = demand.loc[subtour[0]]["time"]
            origin = int(demand.loc[subtour[0]]["origin"])
            destination = int(demand.loc[subtour[-1]]["destination"])
            tour_revenue = 0
            count = 1
            duration = 0
            total_energy_comp = 0
            num_pax = 0
            for i in subtour:
                origin_i = int(demand.loc[i]["origin"])
                destination_i = int(demand.loc[i]["destination"])
                duration += self.flight_time[origin_i, destination_i]
                total_energy_comp += self.energy_consumption[origin_i, destination_i]
                tour_revenue += demand.loc[i]["per_flight_revenue"]
                tour_sequence = (
                    tour_sequence + self.network.vertiport_dict_inv[origin_i] + "-"
                )
                num_pax += demand.loc[i]["per_flight_num_pax"]
            tour_sequence = tour_sequence + self.network.vertiport_dict_inv[destination]

            subtour = pd.DataFrame(
                {
                    "time": [time],
                    "origin": [origin],
                    "destination": [destination],
                    "count": [count],
                    "duration": [duration],
                    "energy_consumption": [total_energy_comp],
                    "num_pax": [num_pax],
                    "tour_revenue": [tour_revenue],
                    "tour_length": [len(subtour)],
                    "tour_sequence": [tour_sequence],
                }
            )

            combined_nodes = pd.concat([combined_nodes, subtour])

        demand["tour_sequence"] = demand.apply(
            lambda row: self.network.vertiport_dict_inv[row["origin"]]
            + "-"
            + self.network.vertiport_dict_inv[row["destination"]],
            axis=1,
        )
        demand["duration"] = demand.apply(
            lambda x: self.flight_time[int(x["origin"]), int(x["destination"])], axis=1
        )
        demand["energy_consumption"] = demand.apply(
            lambda x: self.energy_consumption[int(x["origin"]), int(x["destination"])],
            axis=1,
        )
        demand["tour_revenue"] = demand.apply(lambda x: x["per_flight_revenue"], axis=1)
        demand["num_pax"] = demand.apply(lambda x: x["per_flight_num_pax"], axis=1)

        for idx in selected_lists:
            subtour = list_subtours[idx]
            for i in subtour:
                if demand.loc[i, "count"] == 1:
                    demand.drop(i, inplace=True)
                else:
                    demand.loc[i, "count"] -= 1

        demand = demand.loc[demand.index.repeat(demand["count"])].reset_index(drop=True)
        demand["count"] = 1
        demand["tour_length"] = 1
        output = (
            pd.concat([demand, combined_nodes])
            .sort_values("time")
            .reset_index(drop=True)
        )
        return output

    def _SolveClusteringProblem(self, demand2):
        all_tasks = []
        idx = 0
        for idx, row in demand2.iterrows():
            name = idx
            start_time = row["time"]
            origin = int(row["origin"])
            destination = int(row["destination"])
            duration = row["duration"]
            num_flights = row["tour_length"]
            flight_energy_consumption = row["energy_consumption"]

            all_tasks.append(
                FlightTask(
                    name,
                    start_time,
                    duration,
                    flight_energy_consumption,
                    num_flights,
                    origin,
                    destination,
                    self.flight_time,
                    self.energy_consumption,
                    self.max_zero_prep_time,
                    self.beta_0_coefficient,
                )
            )
            idx += 1

        all_tasks = sorted(all_tasks, key=lambda x: x.start_time)

        network = AssignmentNetwork(all_tasks, self.num_vehicles)
        nodes, supply, edges, cost, c = network.populate_network()

        m = Model("MCNF")

        flow = m.addVars(edges, name="flow", lb=0, ub=1)
        m.setObjective(flow.prod(c), GRB.MAXIMIZE)
        if self.verbose:
            for n in tqdm(nodes):
                m.addConstr(
                    sum(flow[i, j] for i, j in edges if j == n)
                    - sum(flow[i, j] for i, j in edges if i == n)
                    == supply.get(n, 0),
                    f"node_{n}",
                )
        else:
            for n in nodes:
                m.addConstr(
                    sum(flow[i, j] for i, j in edges if j == n)
                    - sum(flow[i, j] for i, j in edges if i == n)
                    == supply.get(n, 0),
                    f"node_{n}",
                )

        m.update()
        m.setParam("OutputFlag", 0)
        m.optimize()
        cluster_id = []
        decisions = []
        for var in m.getVars():
            if var.x > 0:
                decisions.append(var.varName)

        table_data = [self.split_flow_string(flow_str) for flow_str in decisions]
        df = pd.DataFrame(table_data, columns=["Source", "Target"])
        df["source"] = df["Source"].apply(
            lambda x: re.search(r"\d+", x).group() if x != "Source" else None
        )
        df["target"] = df["Target"].apply(
            lambda x: re.search(r"\d+", x).group() if x[-4:] != "Sink" else None
        )
        df = df[df["source"] != df["target"]].reset_index()

        origin_aircraft = df[df["source"].isna()]["target"].values
        list_of_clusters = []
        for idx, node in enumerate(origin_aircraft):
            target = node
            cluster = []
            while target in df["source"].values:
                try:
                    target = df[df["source"] == node]["target"].values[0]
                except IndexError:
                    break
                cluster.append(node)
                node = target
            list_of_clusters.append(cluster)
        cluster_labels = {}
        for cluster_id, cluster in enumerate(list_of_clusters):
            for flight in cluster:
                cluster_labels[flight] = cluster_id

        demand2["cluster_id"] = demand2.index.map(
            lambda x: cluster_labels.get(str(x), -1)
        )

        return demand2

    def _process_cluster(self, cluster_id, demand2):
        unserved_tours = pd.DataFrame(
            columns=[
                "time",
                "origin",
                "destination",
                "count",
                "duration",
                "energy_consumption",
                "num_pax",
                "tour_length",
                "cluster_id",
                "tour_revenue",
            ]
        )
        served_tours = pd.DataFrame(
            columns=[
                "time",
                "origin",
                "destination",
                "count",
                "duration",
                "energy_consumption",
                "num_pax",
                "tour_length",
                "cluster_id",
                "tour_revenue",
            ]
        )

        input_to_vrp = demand2[demand2["cluster_id"] == cluster_id].reset_index(
            drop=True
        )
        m, idx_not_served = self._RunVRP(input_to_vrp)

        # Update unserved tours
        unserved_tours = (
            pd.concat([unserved_tours, input_to_vrp.loc[idx_not_served]])
            .sort_values("time")
            .reset_index(drop=True)
        )

        # Identify served tours and their 'p' and 'q' variables
        idx_served = [i for i in range(len(input_to_vrp)) if i not in idx_not_served]

        # Update served tours
        served_df = input_to_vrp.loc[idx_served].copy()

        served_tours = (
            pd.concat([served_tours, served_df])
            .sort_values("time")
            .reset_index(drop=True)
        )

        return unserved_tours, served_tours

    def _run_parallel_vrp(self, demand2, n_processes):
        pool = multiprocessing.Pool(processes=n_processes)
        results = [
            pool.apply_async(self._process_cluster, args=(i, demand2))
            for i in range(self.num_vehicles)
        ]

        all_unserved = pd.DataFrame(
            columns=[
                "time",
                "origin",
                "destination",
                "count",
                "duration",
                "energy_consumption",
                "num_pax",
                "tour_length",
                "cluster_id",
                "tour_revenue",
            ]
        )
        all_served = pd.DataFrame(
            columns=[
                "time",
                "origin",
                "destination",
                "count",
                "duration",
                "energy_consumption",
                "num_pax",
                "tour_length",
                "cluster_id",
                "tour_revenue",
            ]
        )

        for result in results:
            (
                unserved_tours,
                served_tours,
            ) = result.get()  # Get the result from the process
            all_unserved = (
                pd.concat([all_unserved, unserved_tours])
                .sort_values("time")
                .reset_index(drop=True)
            )
            all_served = (
                pd.concat([all_served, served_tours])
                .sort_values("time")
                .reset_index(drop=True)
            )

        pool.close()
        pool.join()

        return all_unserved, all_served

    def _migrate(self, target_cluster, unserved_tours, served_tours):
        unserved_tours = unserved_tours.copy()
        served_tours = served_tours.copy()

        unserved_tours["served"] = 0
        served_tours["served"] = 1

        input_to_vrp = served_tours[
            served_tours["cluster_id"] == target_cluster
        ].reset_index(drop=True)
        served_tours = served_tours[
            served_tours["cluster_id"] != target_cluster
        ].reset_index(drop=True)

        # commented out the sampling because the previous issue was addressed by relaxing the integer feasibility condition
        # if unserved_tours.shape[0] > 120:
        #     unserved_tours = unserved_tours.sample(n=100)

        input_to_vrp = (
            pd.concat([input_to_vrp, unserved_tours])
            .sort_values("time")
            .reset_index(drop=True)
        )
        m, idx_not_served = self._RunVRP(input_to_vrp, resolve=True)
        idx_served = [i for i in range(len(input_to_vrp)) if i not in idx_not_served]
        p, q, p2, q2 = [], [], [], []
        for idx in idx_served:
            p.append(m.getVarByName(f"p[{idx+1},0]").x)
            q.append(m.getVarByName(f"q[{idx+1},0]").x)
            p2.append(m.getVarByName(f"p2[{idx+1},0]").x)
            q2.append(m.getVarByName(f"q2[{idx+1},0]").x)
        served_df = input_to_vrp.loc[idx_served].copy()
        served_df["cluster_id"] = target_cluster
        served_df["p"] = p
        served_df["q"] = q
        served_df["p2"] = p2
        served_df["q2"] = q2
        served_tours = (
            pd.concat([served_tours, served_df])
            .sort_values("time")
            .reset_index(drop=True)
        )
        unserved_tours = input_to_vrp.loc[idx_not_served].copy().reset_index(drop=True)

        return unserved_tours, served_tours

    def _crossover(self, unserved_tours, served_tours):
        for i in range(self.num_vehicles):
            unserved_tours, served_tours = self._migrate(
                i, unserved_tours, served_tours
            )
        return unserved_tours, served_tours

    # ------------------------------------------------------------------
    # ALNS: Adaptive Large Neighborhood Search (addresses Reviewer 1 #2)
    # Uses random removal + time-related removal as destroy operators,
    # greedy insertion + regret-2 insertion as repair operators.
    # Acceptance criterion: simulated annealing with geometric cooling.
    # ------------------------------------------------------------------

    def _alns(self, unserved_tours, served_tours, n_iter=30):
        """
        ALNS post-processing loop. Iteratively destroys part of the current
        solution and repairs it, accepting improvements (and occasionally
        worse solutions) via simulated annealing.
        """
        import random

        rng = random.Random(42)

        # Operator weights: [random_removal, related_removal] x [greedy, regret2]
        destroy_weights = [1.0, 1.0]
        repair_weights = [1.0, 1.0]
        destroy_ops = [self._random_removal, self._related_removal]
        repair_ops = [self._greedy_insert, self._regret2_insert]

        # Score multipliers for weight update
        SCORE_BEST = 3.0
        SCORE_IMPROVE = 2.0
        SCORE_ACCEPT = 1.0
        WEIGHT_DECAY = 0.8

        best_served = served_tours.copy()
        best_unserved = unserved_tours.copy()
        best_obj = best_served["tour_revenue"].sum()

        current_served = served_tours.copy()
        current_unserved = unserved_tours.copy()
        current_obj = best_obj

        # Simulated annealing parameters
        T = max(best_obj * 0.05, 1.0)
        cooling = math.exp(math.log(0.01) / max(n_iter, 1))

        n_remove = max(1, min(5, len(best_served) // 4))

        for iteration in range(n_iter):
            # Select operators via roulette wheel
            d_idx = self._roulette(rng, destroy_weights)
            r_idx = self._roulette(rng, repair_weights)

            # Destroy
            removed, candidate_served = destroy_ops[d_idx](
                current_served.copy(), current_unserved.copy(), n_remove, rng
            )

            # Repair
            candidate_unserved, candidate_served = repair_ops[r_idx](
                removed, candidate_served
            )

            candidate_obj = candidate_served["tour_revenue"].sum()
            delta = candidate_obj - current_obj

            # Acceptance (simulated annealing)
            if delta >= 0 or rng.random() < math.exp(delta / T):
                current_served = candidate_served
                current_unserved = candidate_unserved
                current_obj = candidate_obj
                score = SCORE_IMPROVE if delta > 0 else SCORE_ACCEPT

                if current_obj > best_obj:
                    best_served = current_served.copy()
                    best_unserved = current_unserved.copy()
                    best_obj = current_obj
                    score = SCORE_BEST
            else:
                score = 0.0

            # Update weights
            destroy_weights[d_idx] = (
                WEIGHT_DECAY * destroy_weights[d_idx] + (1 - WEIGHT_DECAY) * score
            )
            repair_weights[r_idx] = (
                WEIGHT_DECAY * repair_weights[r_idx] + (1 - WEIGHT_DECAY) * score
            )

            T *= cooling

        return best_unserved, best_served

    @staticmethod
    def _roulette(rng, weights):
        """Roulette-wheel selection over operator weights."""
        total = sum(weights)
        r = rng.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1

    def _random_removal(self, served, unserved, n_remove, rng):
        """Destroy operator: randomly remove n_remove tours from served."""
        if served.empty:
            return unserved, served
        indices = rng.sample(range(len(served)), min(n_remove, len(served)))
        removed = served.iloc[indices].copy()
        remaining = served.drop(served.index[indices]).reset_index(drop=True)
        all_unserved = (
            pd.concat([unserved, removed]).sort_values("time").reset_index(drop=True)
        )
        return all_unserved, remaining

    def _related_removal(self, served, unserved, n_remove, rng):
        """
        Destroy operator: remove a seed tour and its n_remove-1 most
        temporally related neighbours (similar departure time).
        """
        if served.empty:
            return unserved, served
        seed_idx = rng.randint(0, len(served) - 1)
        seed_time = served.iloc[seed_idx]["time"]
        time_diffs = (served["time"] - seed_time).abs()
        closest = time_diffs.nsmallest(min(n_remove, len(served))).index
        removed = served.loc[closest].copy()
        remaining = served.drop(closest).reset_index(drop=True)
        all_unserved = (
            pd.concat([unserved, removed]).sort_values("time").reset_index(drop=True)
        )
        return all_unserved, remaining

    def _greedy_insert(self, unserved, served):
        """
        Repair operator: for each vehicle cluster in turn, call _migrate to
        greedily insert unserved tours. Processes vehicles in order.
        """
        for vehicle_id in range(self.num_vehicles):
            if unserved.empty:
                break
            unserved, served = self._migrate(vehicle_id, unserved, served)
        return unserved, served

    def _regret2_insert(self, unserved, served):
        """
        Repair operator: regret-2 insertion. For each unserved tour, estimate
        the gain from inserting into the best vs second-best vehicle cluster.
        Insert the tour with the highest regret (largest gap) first.
        """
        if unserved.empty:
            return unserved, served

        # Compute a simple insertion gain proxy: tour_revenue weighted by
        # how much spare time the cluster has (avoids a full MIP per candidate).
        vehicle_ids = list(range(self.num_vehicles))

        def _cluster_end_time(vid):
            cluster_tours = served[served["cluster_id"] == vid]
            if cluster_tours.empty:
                return 0
            return (cluster_tours["time"] + cluster_tours["duration"]).max()

        remaining_unserved = unserved.copy()
        while not remaining_unserved.empty:
            gains = []
            for _, row in remaining_unserved.iterrows():
                tour_time = row["time"]
                tour_rev = row["tour_revenue"]
                vehicle_gains = []
                for vid in vehicle_ids:
                    end_t = _cluster_end_time(vid)
                    slack = max(0, tour_time - end_t)
                    vehicle_gains.append(tour_rev * (1 + slack / (tour_time + 1)))
                vehicle_gains_sorted = sorted(vehicle_gains, reverse=True)
                regret = (
                    vehicle_gains_sorted[0] - vehicle_gains_sorted[1]
                    if len(vehicle_gains_sorted) >= 2
                    else vehicle_gains_sorted[0]
                )
                gains.append((regret, row.name))

            # Insert tour with highest regret first
            gains.sort(reverse=True)
            best_tour_idx = gains[0][1]
            best_tour = remaining_unserved.loc[[best_tour_idx]]

            # Find best vehicle (highest gain) for this tour
            tour_time = best_tour.iloc[0]["time"]
            best_vid = min(
                vehicle_ids, key=lambda v: abs(_cluster_end_time(v) - tour_time)
            )
            remaining_unserved_temp = remaining_unserved.drop(best_tour_idx)
            unserved_attempt = pd.concat([best_tour, remaining_unserved_temp]).sort_values("time").reset_index(drop=True)

            unserved_attempt, served = self._migrate(best_vid, unserved_attempt, served)

            # Check if it was inserted
            if len(unserved_attempt) < len(remaining_unserved):
                remaining_unserved = unserved_attempt
            else:
                remaining_unserved = remaining_unserved.drop(best_tour_idx).reset_index(drop=True)

        return remaining_unserved, served

    def split_flow_string(self, flow_str):
        open_bracket = flow_str.index("[")
        close_bracket = flow_str.index("]")
        content = flow_str[open_bracket + 1 : close_bracket]
        source, destination = content.split(",", 1)
        return source.strip(), destination.strip()

    def _RunVRP(self, input_to_vrp, K=1, resolve=False, time_limit=1800, optimality_gap=0.1):
        """
        Solve the single- or multi-vehicle eUAMVRP-NL with nonlinear charging.

        Parameters
        ----------
        K                : number of vehicles (1 for CFRS heuristic, >1 for exact benchmark)
        resolve          : if True, lock previously served tours and attempt to insert unserved ones
        time_limit       : Gurobi time limit in seconds
        optimality_gap   : Gurobi MIPGap
        """
        N = input_to_vrp.shape[0] + 2
        if self.linear:
            cs = np.array([0] + [4.3085625] * 32)
        else:
            cs = (
                np.array(
                    [
                        0,
                        0.0129,
                        0.0133,
                        0.0137,
                        0.0142,
                        0.0147,
                        0.0153,
                        0.0158,
                        0.0166,
                        0.0172,
                        0.018,
                        0.0188,
                        0.0197,
                        0.0207,
                        0.0219,
                        0.0231,
                        0.0245,
                        0.026,
                        0.0278,
                        0.03,
                        0.0323,
                        0.0351,
                        0.0384,
                        0.0423,
                        0.0472,
                        0.0536,
                        0.0617,
                        0.0726,
                        0.0887,
                        0.1136,
                        0.1582,
                        0.2622,
                        0.9278,
                    ]
                )
                * 60
            )
        cs = np.cumsum(cs)
        S = np.arange(cs.shape[0])

        alpha_s = np.arange(20, 102.5, 2.5)
        M = 2000

        d = input_to_vrp["time"].values
        d = np.insert(d, 0, 0)
        d = np.append(d, 1600)

        tour_revenue = input_to_vrp["tour_revenue"].values
        tour_revenue = np.insert(tour_revenue, 0, 0)
        tour_revenue = np.append(tour_revenue, 0)

        t = [0]
        gamma = [0]
        for idx, row in input_to_vrp.iterrows():
            t.append(row["duration"])
            gamma.append(row["energy_consumption"])
        t.append(0)
        gamma.append(0)
        t = np.array(t)
        gamma = np.array(gamma)

        tilt = np.zeros((N, N))
        tilgamma = np.zeros((N, N))

        for idx, row in input_to_vrp.iterrows():
            for idx2, row2 in input_to_vrp.iterrows():
                if row["destination"] == row2["origin"]:
                    tilt[idx + 1, idx2 + 1] = 0
                    tilgamma[idx + 1, idx2 + 1] = 0
                else:
                    destination_of_first = row["destination"]
                    origin_of_second = row2["origin"]
                    tilt[idx + 1, idx2 + 1] = self.flight_time[
                        destination_of_first, origin_of_second
                    ]
                    tilgamma[idx + 1, idx2 + 1] = self.energy_consumption[
                        destination_of_first, origin_of_second
                    ]
        # convert repositioning flight back to minutes from time steps
        tilt = tilt * 5

        demand_level = input_to_vrp["count"].values

        m = Model("eVRP")
        x_ijk = [
            (i, j, k) for i in range(N) for j in range(N) for k in range(K) if i < j
        ]
        x_ijk = m.addVars(x_ijk, vtype=GRB.BINARY, name="x")

        alpha_isk = [(i, s, k) for i in range(N) for s in S for k in range(K)]
        alpha_isk = m.addVars(alpha_isk, vtype=GRB.BINARY, name="alpha")

        beta_isk = [(i, s, k) for i in range(N) for s in S for k in range(K)]
        beta_isk = m.addVars(beta_isk, vtype=GRB.BINARY, name="beta")

        p_ik = [(i, k) for i in range(N) for k in range(K)]
        p_ik = m.addVars(p_ik, vtype=GRB.CONTINUOUS, name="p")

        q_ik = [(i, k) for i in range(N) for k in range(K)]
        q_ik = m.addVars(q_ik, vtype=GRB.CONTINUOUS, name="q")

        w_ik = [(i, k) for i in range(N) for k in range(K)]
        w_ik = m.addVars(w_ik, vtype=GRB.CONTINUOUS, name="w")

        v_ik = [(i, k) for i in range(N) for k in range(K)]
        v_ik = m.addVars(v_ik, vtype=GRB.CONTINUOUS, name="v")

        m_isk = [(i, s, k) for i in range(N) for s in S for k in range(K)]
        m_isk = m.addVars(m_isk, vtype=GRB.BINARY, name="m")

        n_isk = [(i, s, k) for i in range(N) for s in S for k in range(K)]
        n_isk = m.addVars(n_isk, vtype=GRB.BINARY, name="n")

        # Charging after repo
        alpha_isk2 = [(i, s, k) for i in range(N) for s in S for k in range(K)]
        alpha_isk2 = m.addVars(alpha_isk2, vtype=GRB.BINARY, name="alpha2")

        beta_isk2 = [(i, s, k) for i in range(N) for s in S for k in range(K)]
        beta_isk2 = m.addVars(beta_isk2, vtype=GRB.BINARY, name="beta2")

        p_ik2 = [(i, k) for i in range(N) for k in range(K)]
        p_ik2 = m.addVars(p_ik2, vtype=GRB.CONTINUOUS, name="p2")

        q_ik2 = [(i, k) for i in range(N) for k in range(K)]
        q_ik2 = m.addVars(q_ik2, vtype=GRB.CONTINUOUS, name="q2")

        w_ik2 = [(i, k) for i in range(N) for k in range(K)]
        w_ik2 = m.addVars(w_ik2, vtype=GRB.CONTINUOUS, name="w2")

        v_ik2 = [(i, k) for i in range(N) for k in range(K)]
        v_ik2 = m.addVars(v_ik2, vtype=GRB.CONTINUOUS, name="v2")

        m_isk2 = [(i, s, k) for i in range(N) for s in S for k in range(K)]
        m_isk2 = m.addVars(m_isk2, vtype=GRB.BINARY, name="m2")

        n_isk2 = [(i, s, k) for i in range(N) for s in S for k in range(K)]
        n_isk2 = m.addVars(n_isk2, vtype=GRB.BINARY, name="n2")

        s_ik = [(i, k) for i in range(N) for k in range(K)]
        s_ik = m.addVars(s_ik, vtype=GRB.CONTINUOUS, name="s")

        # Delta_ik (charging time before repo) and delta_ik (charging time after repo) are
        # directly determined by v_ik - w_ik and v_ik2 - w_ik2, so we inline them rather
        # than carrying redundant auxiliary variables (addresses Reviewer 2 comment 3.2).

        g_ik = [(i, k) for i in range(N) for k in range(K)]
        g_ik = m.addVars(g_ik, vtype=GRB.CONTINUOUS, name="g")

        m.setObjective(
            sum(
                tour_revenue[i] * x_ijk[i, j, k]
                for i in range(1, N - 1)
                for j in range(N)
                for k in range(K)
                if i < j
            ),
            GRB.MAXIMIZE,
        )

        for i in range(1, N - 1):
            m.addConstr(x_ijk.sum(i, "*", "*") <= demand_level[i - 1])  # CONSTRAINT (1)
        if resolve == True:
            served = input_to_vrp["served"].values
            for i in range(1, N - 1):
                if served[i - 1] == 1:
                    m.addConstr(x_ijk.sum(i, "*", "*") == 1)

            # for i in range(1, N-1):
            #     m.addConstr(x_ijk.sum(i, '*', '*') >= served[i-1])

        for k in range(K):
            m.addConstr(x_ijk.sum(0, "*", k) == 1)  # CONSTRAINT (2)
            m.addConstr(x_ijk.sum("*", N - 1, k) == 1)  # CONSTRAINT (3)

        for i in range(1, N - 1):
            for k in range(K):
                m.addConstr(
                    sum(x_ijk[j, i, k] for j in range(N - 1) if j < i)
                    - sum(x_ijk[i, j, k] for j in range(1, N) if i < j)
                    == 0
                )  # CONSTRAINT (4)

        for k in range(K):
            for i in range(N - 1):
                # Before charging
                m.addConstr(
                    p_ik[i, k] == sum(alpha_isk[i, s, k] * alpha_s[s] for s in S)
                )  # CONSTRAINT (5)
                m.addConstr(
                    w_ik[i, k] == sum(alpha_isk[i, s, k] * cs[s] for s in S)
                )  # CONSTRAINT (6)
                m.addConstr(
                    alpha_isk.sum(i, "*", k) == sum(m_isk[i, s, k] for s in S if s != 0)
                )  # CONSTRAINT (7)
                m.addConstr(
                    sum(m_isk[i, s, k] for s in S if s != 0)
                    == sum(x_ijk[i, j, k] for j in range(N) if i < j)
                )  # CONSTRAINT (8)
                m.addConstr(alpha_isk[i, 0, k] <= m_isk[i, 1, k])  # CONSTRAINT (9)
                m.addConstr(
                    alpha_isk[i, max(S), k] <= m_isk[i, max(S), k]
                )  # CONSTRAINT (11)

                # After charging
                m.addConstr(
                    q_ik[i, k] == sum(beta_isk[i, s, k] * alpha_s[s] for s in S)
                )  # CONSTRAINT (12)
                m.addConstr(
                    v_ik[i, k] == sum(beta_isk[i, s, k] * cs[s] for s in S)
                )  # CONSTRAINT (13)
                m.addConstr(
                    beta_isk.sum(i, "*", k) == sum(n_isk[i, s, k] for s in S if s != 0)
                )  # CONSTRAINT (14)
                m.addConstr(
                    sum(n_isk[i, s, k] for s in S if s != 0)
                    == sum(x_ijk[i, j, k] for j in range(N) if i < j)
                )  # CONSTRAINT (15)
                m.addConstr(beta_isk[i, 0, k] <= n_isk[i, 1, k])  # CONSTRAINT (16)
                m.addConstr(
                    beta_isk[i, max(S), k] <= n_isk[i, max(S), k]
                )  # CONSTRAINT (18)

                # Before charging after repo
                m.addConstr(
                    p_ik2[i, k] == sum(alpha_isk2[i, s, k] * alpha_s[s] for s in S)
                )  # CONSTRAINT (19)
                m.addConstr(
                    w_ik2[i, k] == sum(alpha_isk2[i, s, k] * cs[s] for s in S)
                )  # CONSTRAINT (20)
                m.addConstr(
                    alpha_isk2.sum(i, "*", k)
                    == sum(m_isk2[i, s, k] for s in S if s != 0)
                )  # CONSTRAINT (21)
                m.addConstr(
                    sum(m_isk2[i, s, k] for s in S if s != 0)
                    == sum(x_ijk[i, j, k] for j in range(N) if i < j)
                )  # CONSTRAINT (22)
                m.addConstr(alpha_isk2[i, 0, k] <= m_isk2[i, 1, k])  # CONSTRAINT (23)
                m.addConstr(
                    alpha_isk2[i, max(S), k] <= m_isk2[i, max(S), k]
                )  # CONSTRAINT (25)

                # After charging after repo
                m.addConstr(
                    q_ik2[i, k] == sum(beta_isk2[i, s, k] * alpha_s[s] for s in S)
                )  # CONSTRAINT (26)
                m.addConstr(
                    v_ik2[i, k] == sum(beta_isk2[i, s, k] * cs[s] for s in S)
                )  # CONSTRAINT (27)
                m.addConstr(
                    beta_isk2.sum(i, "*", k)
                    == sum(n_isk2[i, s, k] for s in S if s != 0)
                )  # CONSTRAINT (28)
                m.addConstr(
                    sum(n_isk2[i, s, k] for s in S if s != 0)
                    == sum(x_ijk[i, j, k] for j in range(N) if i < j)
                )  # CONSTRAINT (29)
                m.addConstr(beta_isk2[i, 0, k] <= n_isk2[i, 1, k])  # CONSTRAINT (30)
                m.addConstr(
                    beta_isk2[i, max(S), k] <= n_isk2[i, max(S), k]
                )  # CONSTRAINT (32)

                # SoC contiguity: if SoC level s-1 is active but s+1 is not yet reached,
                # level s must also be active — prevents non-contiguous SoC selection
                # (addresses Reviewer 2 comment 3.3).
                for s in range(1, len(S) - 1):
                    m.addConstr(
                        alpha_isk[i, s, k] <= m_isk[i, s, k] + m_isk[i, s + 1, k]
                    )  # CONSTRAINT (10)
                    m.addConstr(
                        beta_isk[i, s, k] <= n_isk[i, s, k] + n_isk[i, s + 1, k]
                    )  # CONSTRAINT (17)
                    m.addConstr(
                        alpha_isk2[i, s, k] <= m_isk2[i, s, k] + m_isk2[i, s + 1, k]
                    )  # CONSTRAINT (24)
                    m.addConstr(
                        beta_isk2[i, s, k] <= n_isk2[i, s, k] + n_isk2[i, s + 1, k]
                    )  # CONSTRAINT (31)
                    # Contiguity: m_isk[s] >= m_isk[s-1] - m_isk[s+1] ensures no gap
                    m.addConstr(m_isk[i, s, k] >= m_isk[i, s - 1, k] - m_isk[i, s + 1, k])
                    m.addConstr(n_isk[i, s, k] >= n_isk[i, s - 1, k] - n_isk[i, s + 1, k])
                    m.addConstr(m_isk2[i, s, k] >= m_isk2[i, s - 1, k] - m_isk2[i, s + 1, k])
                    m.addConstr(n_isk2[i, s, k] >= n_isk2[i, s - 1, k] - n_isk2[i, s + 1, k])

        for k in range(K):
            for i in range(N):
                m.addConstr(g_ik[i, k] == d[i] + t[i])  # CONSTRAINT (35)
                for j in range(N):
                    if i < j:
                        # Delta_ik and delta_ik inlined as (v-w) expressions (no aux var needed)
                        m.addConstr(
                            g_ik[i, k]
                            + (v_ik[i, k] - w_ik[i, k])
                            + t[j]
                            + tilt[i, j]
                            + (v_ik2[i, k] - w_ik2[i, k])
                            - g_ik[j, k]
                            <= M * (1 - x_ijk[i, j, k])
                        )  # CONSTRAINT (36)
                        # Zero-repo: if repositioning time is 0, no second charging session
                        # can occur on edge (i,j) (addresses Reviewer 2 comment 3.1).
                        if tilt[i, j] == 0:
                            m.addConstr(
                                v_ik2[i, k] - w_ik2[i, k]
                                <= M * (1 - x_ijk[i, j, k])
                            )  # second charging time forced to 0 when repo time = 0

        for k in range(K):
            m.addConstr(p_ik2[0, k] == 100)  # CONSTRAINT (37)
            m.addConstr(q_ik2[0, k] == 100)  # CONSTRAINT (38)

        # for i in range(1, N-1):
        #     m.addConstr(p_ik[i, k] >= 0.2) # CONSTRAINT (22)

        for k in range(K):
            for i in range(N):
                for j in range(N):
                    if i < j:
                        m.addConstr(
                            q_ik[i, k]
                            - gamma[j]
                            - tilgamma[i, j]
                            - p_ik[j, k]
                            + q_ik2[i, k]
                            - p_ik2[i, k]
                            <= M * (1 - x_ijk[i, j, k])
                        )  # CONSTRAINT (39)
                        m.addConstr(
                            q_ik[i, k]
                            - gamma[j]
                            - tilgamma[i, j]
                            - p_ik[j, k]
                            + q_ik2[i, k]
                            - p_ik2[i, k]
                            >= -M * (1 - x_ijk[i, j, k])
                        )  # CONSTRAINT (40)
                        m.addConstr(
                            q_ik[i, k] - tilgamma[i, j] - p_ik2[i, k]
                            <= M * (1 - x_ijk[i, j, k])
                        )  # CONSTRAINT (41)
                        m.addConstr(
                            q_ik[i, k] - tilgamma[i, j] - p_ik2[i, k]
                            >= -M * (1 - x_ijk[i, j, k])
                        )  # CONSTRAINT (42)

        # CONSTRAINT (46) — domain bounds
        for k in range(K):
            for i in range(N):
                m.addConstr(w_ik[i, k] >= 0)
                m.addConstr(v_ik[i, k] >= 0)
                m.addConstr(w_ik2[i, k] >= 0)
                m.addConstr(v_ik2[i, k] >= 0)
                m.addConstr(p_ik[i, k] >= 0)
                m.addConstr(q_ik[i, k] >= 0)

        m.update()
        m.Params.OutputFlag = 0
        m.Params.MIPGap = optimality_gap
        m.Params.TimeLimit = time_limit
        m.Params.threads = 8

        if resolve:
            m.Params.NumericFocus = 3
            m.Params.FeasibilityTol = 1e-2
            m.Params.IntFeasTol = 1e-2
            m.Params.MIPGap = min(optimality_gap, 0.05)
        else:
            m.Params.FeasibilityTol = 1e-6
            m.Params.IntFeasTol = 1e-6

        m.optimize()

        total_tours_served = gp.quicksum(
            x_ijk[i, j, k].x for i in range(1, N - 1) for j in range(1, N) if i < j
        ).getValue()

        num_flights_served = 0
        for var in m.getVars():
            if var.varName[0] == "x" and var.x > 0:
                idx = int(var.VarName.split("[")[1].split("]")[0].split(",")[0]) - 1
                if idx >= 0:
                    num_flights_served += input_to_vrp.loc[idx, "tour_length"]
        if self.verbose:
            print(f"TOURS SERVED: {int(total_tours_served)}/{input_to_vrp.shape[0]}")
            print(
                f"FLIGHTS SERVED: {int(num_flights_served)}/{int(input_to_vrp['tour_length'].sum())}"
            )

        list_of_served = []
        for var in m.getVars():
            if var.varName[0] == "x" and var.x > 0:
                idx_served = (
                    int(re.search(r"\d+", var.VarName.split("_")[0]).group()) - 1
                )
                list_of_served.append(idx_served)

        not_served = [
            i for i in range(input_to_vrp.shape[0]) if i not in list_of_served
        ]

        return m, not_served


class FlightTask:
    def __init__(
        self,
        name,
        start_time,
        duration,
        energy_consumption,
        num_flights,
        origin,
        destination,
        flight_time_matrix,
        energy_consumption_matrix,
        max_zero_prep_time,
        beta_0_coefficient,
    ):
        self.name = name
        self.start_time = start_time
        self.duration = duration
        self.land_time = self.start_time + self.duration
        self.energy_consumption = energy_consumption

        self.flight_time_matrix = flight_time_matrix
        self.energy_consumption_matrix = energy_consumption_matrix

        self.num_flights = num_flights
        self.origin = origin
        self.destination = destination

        self.max_zero_prep_time = max_zero_prep_time
        self.beta_0_coefficient = beta_0_coefficient

    def next_task(self, next_task):
        reposition_time = self.flight_time_matrix[self.destination, next_task.origin]
        total_time_in_flight = self.duration + reposition_time

        next_task_start_time = next_task.start_time

        prep_time = self.beta_0_coefficient * max(
            total_time_in_flight - self.max_zero_prep_time, 0
        )

        ready_time = (
            self.land_time
            + self.flight_time_matrix[self.destination, next_task.origin]
            + prep_time
        )

        repo_energy = self.energy_consumption_matrix[self.destination, next_task.origin]
        if_same_dest = self.destination == next_task.origin

        if ready_time <= next_task_start_time:
            return True, repo_energy, ready_time, if_same_dest
        else:
            return False, None, None, None


class AssignmentNetwork:
    def __init__(self, list_of_tasks, num_vehicles):
        self.list_of_tasks = list_of_tasks
        self.num_vehicles = num_vehicles

    def populate_network(self):
        nodes, supply = self._create_nodes()
        edge, cost, c = self._create_edges()

        return nodes, supply, edge, cost, c

    def _create_nodes(self):
        base_nodes = ["Source", "Sink"]
        assignment_nodes = [
            (task.name, status)
            for task in self.list_of_tasks
            for status in ["start", "finish"]
        ]
        nodes = base_nodes + assignment_nodes
        supply = {"Source": -self.num_vehicles, "Sink": self.num_vehicles}

        return nodes, supply

    def _create_edges(self):
        edge, cost = self._create_basic_edges()
        edge_reassignment, cost_reassignment = self._create_reassignment_edges()

        edge = edge + edge_reassignment
        cost = cost + cost_reassignment

        return edge, cost, dict(zip(edge, cost))

    def _create_basic_edges(self):
        cost = []
        source_to_task = [
            ("Source", (task.name, "start")) for task in self.list_of_tasks
        ]
        cost = cost + [0 for _ in range(len(source_to_task))]
        task_to_sink = [((task.name, "finish"), "Sink") for task in self.list_of_tasks]
        cost = cost + [0 for _ in range(len(task_to_sink))]
        task_to_task = [
            ((task.name, "start"), (task.name, "finish")) for task in self.list_of_tasks
        ]
        cost = cost + [task.num_flights for task in self.list_of_tasks]
        basic_edges = source_to_task + task_to_sink + task_to_task

        return basic_edges, cost

    def _create_reassignment_edges(self):
        reassignment_edges = []
        cost = []
        number_of_tasks = len(self.list_of_tasks)

        for i in range(number_of_tasks):
            for j in range(i + 1, number_of_tasks):
                if_connect, repo_energy, ready_time, if_same_dest = self.list_of_tasks[
                    i
                ].next_task(self.list_of_tasks[j])
                if if_connect:
                    reassignment_edges.append(
                        (
                            (self.list_of_tasks[i].name, "finish"),
                            (self.list_of_tasks[j].name, "start"),
                        )
                    )
                    threshold = (
                        repo_energy
                        + self.list_of_tasks[i].energy_consumption
                        + self.list_of_tasks[j].energy_consumption
                    )
                    time_delta = ready_time - self.list_of_tasks[j].start_time

                    if if_same_dest:
                        cost.append(0)
                    elif threshold > 0.4 and time_delta <= 5:
                        cost.append(-0.002)
                    else:
                        cost.append(-0.001)

        # Uncomment the following lines if you want to initialize cost with zeros
        # cost = [0 for _ in range(len(reassignment_edges))]

        return reassignment_edges, cost
