import os

import numpy as np

from .ScheduleGeneratorJFK import ScheduleGenerator
from .utils.schedule_utils import *
from .utils.visualize import plot_travel_time


class StarNetwork:
    def __init__(
        self,
        vertiport_names: list,
        flight_distance_matrix: np.array,
        flight_time_matrix: np.array,
        energy_consumption_matrix: np.array,
    ):
        """
        param vertiports: List of vertiport names. The first element is the hub vertiport (JFK)
        """
        self.vertiports = vertiport_names
        self.vertiport_dict = {i: idx for idx, i in enumerate(vertiport_names)}
        self.vertiport_dict_inv = {idx: i for idx, i in enumerate(vertiport_names)}
        self.flight_distance_matrix = flight_distance_matrix
        self.flight_time = flight_time_matrix
        self.energy_consumption = energy_consumption_matrix

        self.demand_generator = ScheduleGenerator()

    def load_demand(
        self,
        arrival_rate_by_vertiport,
        auto_regressive_alpha=0,
        max_waiting_time=5,
        occupancy=4,
        fare=3,
        seed=9,
    ):
        np.random.seed(seed)
        (
            schedule,
            pax_arrival_times,
        ) = self.demand_generator.get_one_day(
            arrival_rate_by_vertiport=arrival_rate_by_vertiport,
            auto_regressive_alpha=auto_regressive_alpha,
            max_waiting_time=max_waiting_time,
            occupancy=occupancy,
            fare=fare,
            vertiport_dict=self.vertiport_dict,
            flight_distance_matrix=self.flight_distance_matrix,
        )

        self.schedule = schedule = schedule
        self.pax_arrival_times = pax_arrival_times

    def plot_flight(self, ylim=(0, 25)):
        schedule = self.schedule.copy()
        schedule["hour"] = schedule["schedule"] // 60
        schedule.loc[schedule["hour"] == 24.0, "hour"] = 0

        flight_count = schedule.groupby(["hour", "od"]).size().reset_index(name="count")
        pivot_table = flight_count.pivot_table(
            index="hour", columns="od", values="count", fill_value=0
        )
        flight_count = pivot_table.reset_index().melt(
            id_vars="hour", var_name="od", value_name="count"
        )

        flight_count["origin"] = flight_count["od"].apply(lambda x: x.split("_")[0])
        flight_count["destination"] = flight_count["od"].apply(
            lambda x: x.split("_")[1]
        )
        flight_count = flight_count.fillna(0)

        input_to_viz = np.zeros((len(self.vertiports), len(self.vertiports), 24))
        for idx, row in flight_count.iterrows():
            o = row["origin"]
            d = row["destination"]
            h = int(row["hour"])
            input_to_viz[self.vertiport_dict[o], self.vertiport_dict[d], h] = row[
                "count"
            ]

        fig, ax = plot_travel_time(
            input_to_viz, self.vertiports, ylim=ylim, ylabel="Number of Flights"
        )

        return fig, ax

    def plot_pax(self, ylim=(0, 80)):
        pax_arrival_times = self.pax_arrival_times.copy()
        pax_arrival_times["hour"] = (
            pax_arrival_times["passenger_arrival_time_s"] // 3600
        )
        pax_count = (
            pax_arrival_times.groupby(
                ["origin_vertiport_id", "destination_vertiport_id", "hour"]
            )
            .size()
            .reset_index(name="counts")
        )

        input_to_viz = np.zeros((len(self.vertiports), len(self.vertiports), 24))
        for idx, row in pax_count.iterrows():
            o = int(row["origin_vertiport_id"])
            d = int(row["destination_vertiport_id"])
            h = int(row["hour"])
            input_to_viz[o, d, h] = row["counts"]

        fig, ax = plot_travel_time(
            input_to_viz, self.vertiports, ylim=ylim, ylabel="Number of Passengers"
        )

        return fig, ax
