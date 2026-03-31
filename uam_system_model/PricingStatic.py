import math
import re

import numpy as np
import pandas as pd
from gurobipy import *
from tqdm import tqdm


class PricingOptimizerStatic:
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
        uam_fare,
        value_of_time,
        beta_time=-0.0192,
        beta_cost=-0.0353,
        uam_transition_time=10,
        CASM=0.79,
        verbose=True,
    ):
        if isinstance(value_of_time, int) or isinstance(value_of_time, float):
            value_of_time = [
                value_of_time for _ in range(len(self.network.vertiport_dict))
            ]
        elif len(value_of_time) != len(self.network.vertiport_dict):
            raise ValueError("Length of value_of_time must match number of vertiports")
        if uam_fare.shape != first_or_last_distance.shape:
            raise ValueError(
                "Shapes of uam_fare, first_or_last_distance, and uam_flight_time must match"
            )

        pax_arr = self.network.pax_arrival_times.copy()
        self.time_resolution = time_resolution
        self.num_vehicles = num_vehicles

        self.beta_time = [0] + [
            beta_cost * value_of_time[i] / 60
            for i in range(len(self.network.vertiport_dict))
        ]
        self.beta_cost = [beta_cost for _ in range(len(self.network.vertiport_dict))]

        pax_arr["passenger_arrival_time_slot"] = np.floor(
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
        uam_operating_cost = (
            np.repeat(self.network.flight_distance_matrix[:, :, np.newaxis], 24, axis=2)
            * CASM
            * 4
        )

        uam_cost = np.zeros(
            shape=(
                len(self.network.vertiport_dict),
                len(self.network.vertiport_dict),
                24,
            )
        )
        for o in range(len(self.network.vertiport_dict)):
            for d in range(len(self.network.vertiport_dict)):
                for h in range(24):
                    uam_cost[o, d, h] = (
                        uam_fare[o, d, h] + first_or_last_distance[o, d, h] * 2.5
                    )

        uam_travel_time = np.zeros(
            shape=(
                len(self.network.vertiport_dict),
                len(self.network.vertiport_dict),
                24,
            )
        )
        for o in range(len(self.network.vertiport_dict)):
            for d in range(len(self.network.vertiport_dict)):
                for h in range(24):
                    uam_travel_time[o, d, h] = (
                        uam_flight_time[o, d]
                        + first_mile_time[o, h]
                        + last_mile_time[d, h]
                        + uam_transition_time
                    )

        utility_uam = np.zeros(
            shape=(
                len(self.network.vertiport_dict),
                len(self.network.vertiport_dict),
                24,
            )
        )
        for o in range(len(self.network.vertiport_dict)):
            for d in range(len(self.network.vertiport_dict)):
                for h in range(24):
                    if o != 0:
                        utility_uam[o, d, h] = (
                            self.beta_time[o] * uam_travel_time[o, d, h]
                            + self.beta_cost[o] * uam_cost[o, d, h]
                        )
                    else:
                        utility_uam[o, d, h] = (
                            self.beta_time[d] * uam_travel_time[o, d, h]
                            + self.beta_cost[d] * uam_cost[o, d, h]
                        )
        utility_uber = np.zeros(
            shape=(
                len(self.network.vertiport_dict),
                len(self.network.vertiport_dict),
                24,
            )
        )
        for o in range(len(self.network.vertiport_dict)):
            for d in range(len(self.network.vertiport_dict)):
                for h in range(24):
                    utility_uber[o, d, h] = (
                        self.beta_time[o] * uber_travel_time[o, d, h]
                        + self.beta_cost[o] * uber_fare[o, d, h]
                    )
        market_share_uam = np.zeros(
            shape=(
                len(self.network.vertiport_dict),
                len(self.network.vertiport_dict),
                24,
            )
        )
        for o in range(len(self.network.vertiport_dict)):
            for d in range(len(self.network.vertiport_dict)):
                for h in range(24):
                    exp_uam = np.exp(utility_uam[o, d, h])
                    exp_uber = np.exp(utility_uber[o, d, h])
                    market_share_uam[o, d, h] = (
                        exp_uam / (exp_uam + exp_uber)
                        if (exp_uam + exp_uber) > 0
                        else 0.0
                    )

        self.pax_arr_grouped["market_share_uam"] = self.pax_arr_grouped.apply(
            lambda row: market_share_uam[
                int(row["origin_vertiport_id"]),
                int(row["destination_vertiport_id"]),
                int(row["passenger_arrival_time_slot"] // 2),
            ],
            axis=1,
        )
        self.pax_arr_grouped["uam_pax"] = (
            self.pax_arr_grouped["counts"] * self.pax_arr_grouped["market_share_uam"]
        )
        self.pax_arr_grouped["uam_pax"] = (
            self.pax_arr_grouped["uam_pax"].round().astype(int)
        )

        all_tasks = []
        for idx, row in self.pax_arr_grouped.iterrows():
            o = int(row["origin_vertiport_id"])
            d = int(row["destination_vertiport_id"])
            h = int(row["passenger_arrival_time_slot"] // 2)
            slot = int(row["passenger_arrival_time_slot"])
            count = row["uam_pax"]
            flight_id = 0
            while count > 0:
                num_pax = min(
                    count, 4
                )  # Assuming each flight can take up to 4 passengers
                all_tasks.append(
                    FlightTask(
                        name=f"task_{o}_{d}_{slot}_{flight_id}_{int(num_pax)}",
                        start_time=row["passenger_arrival_time_slot"],
                        uam_distance_matrix=self.network.flight_distance_matrix,
                        duration=self.flight_time_matrix[
                            int(row["origin_vertiport_id"]),
                            int(row["destination_vertiport_id"]),
                        ],
                        origin=int(row["origin_vertiport_id"]),
                        destination=int(row["destination_vertiport_id"]),
                        flight_time_matrix=self.flight_time_matrix,
                        num_pax=num_pax,
                        profit=uam_fare[o, d, h] * num_pax
                        - uam_operating_cost[o, d, h],
                    )
                )
                count -= num_pax
                flight_id += 1

        assignment_network = AssignmentNetwork(
            all_tasks, num_vehicles=num_vehicles, CASM=CASM
        )
        (
            self.nodes,
            self.supply,
            self.edges,
            self.di_bar,
            self.c,
        ) = assignment_network.populate_network()

        m = Model("MCNF")

        flow = m.addVars(self.edges, name="flow", lb=0, ub=1)
        m.setObjective(flow.prod(self.c), GRB.MAXIMIZE)
        if verbose:
            for n in tqdm(self.nodes):
                m.addConstr(
                    sum(flow[i, j] for i, j in self.edges if j == n)
                    - sum(flow[i, j] for i, j in self.edges if i == n)
                    == self.supply.get(n, 0),
                    f"node_{n}",
                )
        else:
            for n in self.nodes:
                m.addConstr(
                    sum(flow[i, j] for i, j in self.edges if j == n)
                    - sum(flow[i, j] for i, j in self.edges if i == n)
                    == self.supply.get(n, 0),
                    f"node_{n}",
                )

        m.update()
        m.setParam("OutputFlag", 0)
        m.optimize()

        decisions = []
        value = []
        for var in m.getVars():
            decisions.append(var.varName)
            value.append(var.x)

        results = pd.DataFrame({"Variable": decisions, "value": value})
        pattern = r"\[\s*\(['\"].*?(\d+)['\"]\s*,\s*['\"]start['\"]\)\s*,\s*\(['\"].*?(\d+)['\"]\s*,\s*['\"]finish['\"]\)\s*\]"
        flights = results[
            results["Variable"].str.contains(pattern, regex=True)
        ].reset_index(drop=True)
        flights = flights.reset_index()
        flights = flights.rename(columns={"index": "task"})

        flights["extracted_nums"] = flights["Variable"].apply(
            self.extract_specific_numbers
        )
        flights["num_pax"] = flights["extracted_nums"].apply(lambda x: int(x[0][3]))
        flights["origin_vertiport_id"] = flights["extracted_nums"].apply(
            lambda x: int(x[0][0])
        )
        flights["destination_vertiport_id"] = flights["extracted_nums"].apply(
            lambda x: int(x[0][1])
        )
        flights["time_slot"] = flights["extracted_nums"].apply(lambda x: int(x[0][2]))
        flights["fare"] = flights.apply(
            lambda row: uam_fare[
                int(row["origin_vertiport_id"]),
                int(row["destination_vertiport_id"]),
                int(row["time_slot"] // 2),
            ],
            axis=1,
        )
        flights["revenue"] = flights.apply(
            lambda row: row["num_pax"] * row["fare"] if row["value"] else 0, axis=1
        )
        flights["cost"] = flights.apply(
            lambda row: uam_operating_cost[
                int(row["origin_vertiport_id"]),
                int(row["destination_vertiport_id"]),
                int(row["time_slot"] // 2),
            ]
            if row["value"]
            else 0,
            axis=1,
        )
        flights["profit"] = flights["revenue"] - flights["cost"]

        pattern = r"\[\s*\(['\"].*?(\d+)['\"]\s*,\s*['\"]finish['\"]\)\s*,\s*\(['\"].*?(\d+)['\"]\s*,\s*['\"]start['\"]\)\s*\]"
        repo_flights = results[
            results["Variable"].str.contains(pattern, regex=True)
        ].reset_index(drop=True)
        repo_flights = repo_flights[repo_flights["value"] == 1].reset_index(drop=True)
        repo_flights["extracted_nums"] = repo_flights["Variable"].apply(
            self.extract_specific_numbers
        )
        repo_flights["repo_origin_vertiport_id"] = repo_flights["extracted_nums"].apply(
            lambda x: int(x[0][1])
        )
        repo_flights["repo_destination_vertiport_id"] = repo_flights[
            "extracted_nums"
        ].apply(lambda x: int(x[1][0]))
        repo_flights["repo_distance"] = repo_flights.apply(
            lambda row: self.network.flight_distance_matrix[
                int(row["repo_origin_vertiport_id"]),
                int(row["repo_destination_vertiport_id"]),
            ],
            axis=1,
        )
        repo_flights["repo_cost"] = repo_flights["repo_distance"] * 4 * CASM

        return flights, repo_flights

    @staticmethod
    def extract_specific_numbers(text):
        parentheses_contents = re.findall(r"\((.*?)\)", str(text))

        result = []
        for content in parentheses_contents:
            nums = re.findall(r"\d+", content)

            if len(nums) >= 3:
                result.append((nums[0], nums[1], nums[2], nums[-1]))
            elif nums:
                result.append(tuple(nums))

        return result


class FlightTask:
    def __init__(
        self,
        name,
        start_time,
        duration,
        origin,
        destination,
        flight_time_matrix,
        uam_distance_matrix,
        num_pax,
        profit,
    ):
        self.name = name
        self.start_time = start_time
        self.duration = duration
        self.land_time = self.start_time + self.duration
        self.flight_time_matrix = flight_time_matrix
        self.uam_distance_matrix = uam_distance_matrix
        self.origin = origin
        self.destination = destination
        self.num_pax = num_pax
        self.profit = profit

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
    def __init__(self, list_of_tasks, num_vehicles, CASM):
        self.list_of_tasks = list_of_tasks
        self.num_vehicles = num_vehicles
        self.CASM = CASM

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
        di_bar = di_bar + [task.profit for task in self.list_of_tasks]
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

                    repo_distance = self.list_of_tasks[i].uam_distance_matrix[
                        self.list_of_tasks[i].destination, self.list_of_tasks[j].origin
                    ]
                    di_bar.append(-1 * repo_distance * 4 * self.CASM)

        return reassignment_edges, di_bar
