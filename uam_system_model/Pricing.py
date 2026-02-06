import math
import re

import numpy as np
import pandas as pd
from gurobipy import *
from tqdm import tqdm


class PricingOptimizer:
    def __init__(
        self,
        StarNetwork,
    ):
        self.network = StarNetwork

    def optimize(
        self,
        time_resolution,
        num_vehicles,
        uber_travel_time,
        uber_fare,
        first_mile_time,
        last_mile_time,
        first_or_last_distance,
        uam_flight_time,
        uam_distance_matrix,
        optimality_gap,
        value_of_time,
        utility_type,
        beta_time=-0.0192,
        beta_cost=-0.0353,
        uam_transition_time=10,
        time_limit=1800,
        CASM=0.79,
        verbose=True,
    ):
        if isinstance(value_of_time, int) or isinstance(value_of_time, float):
            value_of_time = [
                value_of_time for _ in range(len(self.network.vertiport_dict))
            ]

        pax_arr = self.network.pax_arrival_times.copy()
        self.time_resolution = time_resolution
        self.num_vehicles = num_vehicles
        self.beta_time = [
            beta_cost * value_of_time[i] / 60
            for i in range(len(self.network.vertiport_dict))
        ]
        self.beta_cost = [beta_cost for _ in range(len(self.network.vertiport_dict))]

        pax_arr["passenger_arrival_time_slot"] = np.ceil(
            pax_arr["passenger_arrival_time_s"] / self.time_resolution / 60
        ).astype(int)
        pax_arr_grouped = (
            pax_arr.groupby(
                [
                    "origin_vertiport_id",
                    "destination_vertiport_id",
                    "passenger_arrival_time_slot",
                ]
            )
            .size()
            .reset_index(name="counts")
        )
        self.pax_arr_grouped = pax_arr_grouped.sort_values(
            by=["passenger_arrival_time_slot"]
        ).reset_index(drop=True)

        self.flight_time_matrix = np.ceil(
            self.network.flight_time * 5 / self.time_resolution
        )

        all_tasks = []
        for idx, row in self.pax_arr_grouped.iterrows():
            all_tasks.append(
                FlightTask(
                    name=f"Task_{idx}",
                    start_time=row["passenger_arrival_time_slot"],
                    duration=self.flight_time_matrix[
                        row["origin_vertiport_id"], row["destination_vertiport_id"]
                    ],
                    origin=row["origin_vertiport_id"],
                    destination=row["destination_vertiport_id"],
                    flight_time_matrix=self.flight_time_matrix,
                    num_pax=row["counts"],
                )
            )

        assignment_network = AssignmentNetwork(all_tasks, num_vehicles=num_vehicles)
        (
            self.nodes,
            self.supply,
            self.edges,
            self.di_bar,
            self.c,
        ) = assignment_network.populate_network()

        uber_travel_time_i = []
        # uber_fare = np.array([0, 50, 62, 78, 57, 77, 57, 57, 45])
        uber_fare_i = []

        t_i_uam = []
        first_last_mile_cost = []
        flight_cost_uam = []
        beta_time_i = []
        beta_cost_i = []
        p_i_bar = []

        for idx, row in self.pax_arr_grouped.iterrows():
            origin = row["origin_vertiport_id"]
            destination = row["destination_vertiport_id"]
            time = row["passenger_arrival_time_slot"]
            time = np.ceil(time / (60 / self.time_resolution)).astype(int) - 1

            if origin == 0:
                t_i_uam.append(
                    uam_flight_time[origin, destination]
                    + last_mile_time[destination, time]
                    + uam_transition_time
                )

            elif destination == 0:
                t_i_uam.append(
                    first_mile_time[origin, time]
                    + uam_flight_time[origin, destination]
                    + uam_transition_time
                )

            first_last_mile_cost.append(
                first_or_last_distance[origin, destination, time] * 2.5
            )

            uber_travel_time_i.append(uber_travel_time[origin, destination, time])

            if len(uber_fare.shape) == 1:
                od_idx = max(origin, destination)
                uber_fare_i.append(uber_fare[od_idx])
            elif len(uber_fare.shape) == 3:
                od_idx = max(origin, destination)
                uber_fare_i.append(uber_fare[origin, destination, time])

            p_i_bar.append(1 / value_of_time[od_idx])
            beta_time_i.append(self.beta_time[od_idx])
            beta_cost_i.append(self.beta_cost[od_idx])

            distance = uam_distance_matrix[origin, destination]
            flight_cost_uam.append(CASM * 4 * distance)

        uber_travel_time_i = np.array(uber_travel_time_i)
        uber_fare_i = np.array(uber_fare_i)

        t_i_uam = np.array(t_i_uam)

        non_zero_indices = [i for i, value in enumerate(self.di_bar) if value != 0]

        di_bar_selected_x = [self.di_bar[i] for i in non_zero_indices]

        if utility_type == "vot":
            if isinstance(value_of_time, float):
                p_i_bar = [
                    1 / value_of_time for _ in non_zero_indices
                ]  # 32.63 is the VOT in dollars per minute

            else:
                p_i_bar = []
                for idx, row in self.pax_arr_grouped.iterrows():
                    origin = row["origin_vertiport_id"]
                    destination = row["destination_vertiport_id"]
                    od_idx = max(origin, destination)
                    p_i_bar.append(1 / (value_of_time[od_idx] / 60))

            v_i_bar_uber = -uber_travel_time_i - p_i_bar * uber_fare_i
        elif utility_type == "betas":
            beta_time_i = np.array(beta_time_i)
            beta_cost_i = np.array(beta_cost_i)
            v_i_bar_uber = beta_time_i * uber_travel_time_i + beta_cost_i * uber_fare_i

        bins = 20
        max_flights = num_vehicles
        eps = 0.05

        m = Model("Pricing Problem")
        m.Params.NonConvex = 2
        m.setParam("MIPGap", optimality_gap)
        m.setParam("TimeLimit", time_limit)

        m._x_vars = m.addVars(self.edges, vtype=GRB.INTEGER, name="x_ij")
        m._x_inverse_vars = m.addVars(
            self.edges, vtype=GRB.CONTINUOUS, name="x_ij_inverse", lb=0
        )

        m._theta_uam = m.addVars(
            len(non_zero_indices),
            vtype=GRB.CONTINUOUS,
            name="theta_uam",
            lb=0,
            ub=1 - eps,
        )
        m._theta_ln_theta_uam = m.addVars(
            len(non_zero_indices),
            vtype=GRB.CONTINUOUS,
            name="ln_theta_uam",
            lb=-float("inf"),
            ub=0,
        )

        m._theta_ln_1_minus_theta_uam = m.addVars(
            len(non_zero_indices),
            vtype=GRB.CONTINUOUS,
            name="1_minus_ln_theta_uam",
            lb=-float("inf"),
            ub=0,
        )

        xs = [1 / bins * i for i in range(bins + 1)]
        x_inverse_s = [i for i in range(max_flights + 1)]

        ys = [p * math.log(p) if p != 0 else 0 for p in xs]
        ys2 = [p * math.log(1 - p) if p != 1 else 0 for p in xs]

        y_inverse_s = [1 / x if x != 0 else 0 for x in x_inverse_s]

        for i in range(len(non_zero_indices)):
            m.addGenConstrPWL(
                m._theta_uam[i], m._theta_ln_theta_uam[i], xs, ys, name=f"pwl_{i}"
            )
            m.addGenConstrPWL(
                m._theta_uam[i],
                m._theta_ln_1_minus_theta_uam[i],
                xs,
                ys2,
                name=f"pwl2_{i}",
            )

        # Flow conservation constraints
        for n in self.nodes:
            m.addConstr(
                sum(m._x_vars[i, j] for i, j in self.edges if j == n)
                - sum(m._x_vars[i, j] for i, j in self.edges if i == n)
                == self.supply.get(n, 0),
                f"node_{n}",
            )

        # Capacity constraints
        for i in range(len(non_zero_indices)):
            idx = non_zero_indices[i]
            m.addConstr(
                m._x_vars[self.edges[idx]] * 4
                >= m._theta_uam[i] * di_bar_selected_x[i],
                f"cap_{i}",
            )
            m.addGenConstrPWL(
                m._x_vars[self.edges[idx]],
                m._x_inverse_vars[self.edges[idx]],
                x_inverse_s,
                y_inverse_s,
                name=f"inv_pwl_{i}",
            )

        operating_cost = quicksum(
            m._x_vars[self.edges[i]] * flight_cost_uam[i_non_zero]
            for i_non_zero, i in zip(range(len(di_bar_selected_x)), non_zero_indices)
        )
        if utility_type == "vot":
            cost_level_of_service = quicksum(
                di_bar_selected_x[i_non_zero]
                / p_i_bar[i_non_zero]
                * m._x_inverse_vars[self.edges[i]]
                * self.time_resolution
                / 2
                for i_non_zero, i in zip(
                    range(len(di_bar_selected_x)), non_zero_indices
                )
            )
            theta_terms = quicksum(
                di_bar_selected_x[i_non_zero]
                / p_i_bar[i_non_zero]
                * (
                    m._theta_ln_theta_uam[i_non_zero]
                    - m._theta_ln_1_minus_theta_uam[i_non_zero]
                )
                for i_non_zero in range(len(di_bar_selected_x))
            )

            other_terms = quicksum(
                di_bar_selected_x[i_non_zero]
                / p_i_bar[i_non_zero]
                * m._theta_uam[i_non_zero]
                * (v_i_bar_uber[i_non_zero] + t_i_uam[i_non_zero])
                for i_non_zero in range(len(di_bar_selected_x))
            )

            objective = (
                theta_terms + other_terms + operating_cost + cost_level_of_service
            )

        elif utility_type == "betas":
            theta_terms = quicksum(
                -di_bar_selected_x[i_non_zero]
                / beta_cost_i[i_non_zero]
                * (
                    m._theta_ln_theta_uam[i_non_zero]
                    - m._theta_ln_1_minus_theta_uam[i_non_zero]
                )
                for i_non_zero in range(len(di_bar_selected_x))
            )

            other_terms = quicksum(
                -di_bar_selected_x[i_non_zero]
                / beta_cost_i[i_non_zero]
                * m._theta_uam[i_non_zero]
                * (
                    v_i_bar_uber[i_non_zero]
                    - beta_time_i[i_non_zero] * t_i_uam[i_non_zero]
                )
                for i_non_zero in range(len(di_bar_selected_x))
            )

            # objective = theta_terms + other_terms + operating_cost + cost_level_of_service
            objective = theta_terms + other_terms + operating_cost

        m.setObjective(objective, GRB.MINIMIZE)

        if utility_type == "betas":
            max_v = max_flights
            for i, idx in enumerate(non_zero_indices):
                current_los_costs = []
                factor = (
                    di_bar_selected_x[i] * beta_time_i[i] * self.time_resolution
                ) / (2 * beta_cost_i[i])
                for k in range(max_v + 1):
                    if k == 0:
                        current_los_costs.append(1e6)  # Penalty for no service
                    else:
                        current_los_costs.append(factor * (1.0 / k))

                m.setPWLObj(
                    m._x_vars[self.edges[idx]], range(max_v + 1), current_los_costs
                )
        if not verbose:
            m.setParam("OutputFlag", 0)
        m.update()
        m.optimize()
        self.model = m

        results = []
        for v in m.getVars():
            if v.X > 0.01:
                results.append((v.VarName, v.X))
        results = pd.DataFrame(results, columns=["Variable", "Value"])
        prices = results[results["Variable"].str.contains("theta_uam")].reset_index(
            drop=True
        )
        prices["flight_index"] = prices["Variable"].apply(
            lambda x: int(re.findall(r"\d+", x)[0])
        )
        prices["percentage_uam"] = prices["Value"].astype(float)

        prices = prices[["flight_index", "percentage_uam"]]
        flow = results[results["Variable"].str.contains("x_ij")]

        pattern = r"x_ij\[\('Task_\d+', 'start'\)"
        flow = flow[flow["Variable"].str.contains(pattern)].reset_index(drop=True)
        flow["flight_index"] = flow["Variable"].apply(
            lambda x: int(re.findall(r"Task_(\d+)", x)[0])
        )
        flow = flow[["flight_index", "Value"]]
        flow = flow.rename(columns={"Value": "num_flights"})
        output_merged = prices.merge(flow, on="flight_index", how="left")

        self.v_i_bar_uber = v_i_bar_uber
        self.t_i_uam = t_i_uam
        self.value_of_time = value_of_time

        if utility_type == "vot":
            output_merged["fare"] = output_merged.apply(
                lambda row: self.calc_fare_vot(
                    row,
                    1 / p_i_bar[int(row["flight_index"])],
                    v_i_bar_uber,
                    t_i_uam,
                    first_last_mile_cost,
                    self.time_resolution,
                ),
                axis=1,
            )
        elif utility_type == "betas":
            output_merged["fare"] = output_merged.apply(
                lambda row: self.calc_fare_betas(
                    row,
                    beta_time_i,
                    beta_cost_i,
                    v_i_bar_uber,
                    t_i_uam,
                    first_last_mile_cost,
                    self.time_resolution,
                ),
                axis=1,
            )

        pax_arr_grouped_to_join = self.pax_arr_grouped.copy()
        pax_arr_grouped_to_join["flight_index"] = pax_arr_grouped_to_join.index
        pax_arr_grouped_to_join = pax_arr_grouped_to_join.rename(
            columns={"counts": "num_pax"}
        )
        df = pd.merge(
            output_merged, pax_arr_grouped_to_join, on="flight_index", how="left"
        )

        df["distance"] = df.apply(
            lambda row: uam_distance_matrix[
                int(row["origin_vertiport_id"]), int(row["destination_vertiport_id"])
            ],
            axis=1,
        )
        df["uam_pax"] = df["num_pax"] * df["percentage_uam"]

        df["rev_per_mile"] = df["fare"] / df["distance"]

        df["markets"] = df.apply(
            lambda row: int(
                max(row["origin_vertiport_id"], row["destination_vertiport_id"])
            ),
            axis=1,
        )
        df["markets"] = df["markets"].apply(
            lambda x: self.network.vertiport_dict_inv[x]
        )

        df["rev_per_mile"] = df["fare"] / df["distance"]
        df["total_revenue"] = df["fare"] * df["uam_pax"]

        return df

    @staticmethod
    def calc_fare_vot(
        row, value_of_time, v_i_bar_uber, t_i_uam, first_last_mile_cost, time_resolution
    ):
        num_flights = row["num_flights"]
        theta = row["percentage_uam"]
        index = int(row["flight_index"])
        first_last_cost = first_last_mile_cost[index]

        fare = -value_of_time * (
            math.log(theta)
            - math.log(1 - theta)
            + v_i_bar_uber[index]
            + t_i_uam[index]
            + time_resolution / 2 / num_flights
        )

        return fare - first_last_cost

    @staticmethod
    def calc_fare_betas(
        row,
        beta_time,
        beta_cost,
        v_i_bar_uber,
        t_i_uam,
        first_last_mile_cost,
        time_resolution,
    ):
        num_flights = row["num_flights"]
        theta = row["percentage_uam"]
        index = int(row["flight_index"])
        first_last_cost = first_last_mile_cost[index]
        beta_time = beta_time[index]
        beta_cost = beta_cost[index]

        if theta >= 1:
            theta = 0.9999
        if theta <= 0:
            theta = 0.0001

        fare = (
            1
            / beta_cost
            * (
                math.log(theta)
                - math.log(1 - theta)
                + v_i_bar_uber[index]
                - beta_time * (t_i_uam[index] + time_resolution / 2 / num_flights)
            )
        )

        return fare - first_last_cost


class FlightTask:
    def __init__(
        self,
        name,
        start_time,
        duration,
        origin,
        destination,
        flight_time_matrix,
        num_pax,
    ):
        self.name = name
        self.start_time = start_time
        self.duration = duration
        self.land_time = self.start_time + self.duration
        self.flight_time_matrix = flight_time_matrix
        self.origin = origin
        self.destination = destination
        self.num_pax = num_pax

    def next_task(self, next_task):
        reposition_time = self.flight_time_matrix[self.destination, next_task.origin]

        next_task_start_time = next_task.start_time

        ready_time = (
            self.land_time
            + self.flight_time_matrix[self.destination, next_task.origin]
            + reposition_time
        )

        if (
            next_task_start_time - ready_time >= 0
            and next_task_start_time - ready_time <= 12
        ):
            return True
        else:
            return False


class AssignmentNetwork:
    def __init__(self, list_of_tasks, num_vehicles):
        self.list_of_tasks = list_of_tasks
        self.num_vehicles = num_vehicles

    def populate_network(self):
        nodes, supply = self._create_nodes()
        edge, di_bar, c = self._create_edges()

        return nodes, supply, edge, di_bar, c

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
        edge, di_bar = self._create_basic_edges()
        edge_reassignment, di_bar_reassignment = self._create_reassignment_edges()

        edge = edge + edge_reassignment
        di_bar = di_bar + di_bar_reassignment

        return edge, di_bar, dict(zip(edge, di_bar))

    def _create_basic_edges(self):
        di_bar = []
        source_to_task = [
            ("Source", (task.name, "start")) for task in self.list_of_tasks
        ]
        di_bar = di_bar + [0 for _ in range(len(source_to_task))]
        task_to_sink = [((task.name, "finish"), "Sink") for task in self.list_of_tasks]
        di_bar = di_bar + [0 for _ in range(len(task_to_sink))]
        task_to_task = [
            ((task.name, "start"), (task.name, "finish")) for task in self.list_of_tasks
        ]
        di_bar = di_bar + [task.num_pax for task in self.list_of_tasks]
        basic_edges = source_to_task + task_to_sink + task_to_task

        return basic_edges, di_bar

    def _create_reassignment_edges(self):
        reassignment_edges = []
        di_bar = []
        number_of_tasks = len(self.list_of_tasks)

        for i in range(number_of_tasks):
            for j in range(i + 1, number_of_tasks):
                if_connect = self.list_of_tasks[i].next_task(self.list_of_tasks[j])
                if if_connect:
                    reassignment_edges.append(
                        (
                            (self.list_of_tasks[i].name, "finish"),
                            (self.list_of_tasks[j].name, "start"),
                        )
                    )

                    di_bar.append(0)

        return reassignment_edges, di_bar
