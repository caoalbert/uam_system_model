import os
import numpy as np

DATA_PATH_1 = os.path.join(os.path.dirname(__file__), "data", "2019_ASPM_FLIGHT_LEVEL.csv")
DATA_PATH_2 = os.path.join(os.path.dirname(__file__), "data", "T_F41SCHEDULE_B43.csv")

from .utils.schedule_utils import *
from .ScheduleGenerator import ScheduleGenerator

class StarNetwork:
    def __init__(
        self,
        vertiport_names: list,
        flight_distance_matrix: np.array,
        flight_time_matrix: np.array,
        energy_consumption_matrix: np.array,
        path_schedule: str = DATA_PATH_1,
        path_seat_capacity: str = DATA_PATH_2,
    ):
        """
        param vertiports: List of vertiport names. The first element is the hub vertiport (LAX)

        """
        self.vertiports = vertiport_names
        self.vertiport_dict = {i: idx for idx, i in enumerate(vertiport_names)}
        self.vertiport_dict_inv = {idx: i for idx, i in enumerate(vertiport_names)}
        self.flight_distance_matrix = flight_distance_matrix
        self.flight_time = flight_time_matrix
        self.energy_consumption = energy_consumption_matrix

        self.demand_generator = ScheduleGenerator(path_schedule, path_seat_capacity)

    def load_demand(
        self,
        month: int,
        day: int,
        directional_demand: int,
        vertiport_pmf: np.array,
        auto_regressive_alpha: float = 0,
        max_waiting_time: int = 5,
        occupancy: int = 4,
        seed: int = 9,
        fare: float = 3.0,
    ):
        self.month = month
        self.day = day
        """
        Load the demand for a specific day and month.
        :param month: Month of the year (1-12)
        :param day: Day of the month (1-31)
        :param directional_demand: Total demand for the day
        :return: A tuple containing the schedule, passenger arrival times, and number of passengers per flight.
        """

        if len(self.vertiport_dict) - len(vertiport_pmf):
            raise ValueError(
                "Length of vertiport_names and vertiport_pmf must be the same."
            )

        np.random.seed(seed)

        (
            schedule,
            pax_arrival_times,
        ) = self.demand_generator.get_one_day(
            month=month,
            day=day,
            auto_regressive_alpha=auto_regressive_alpha,
            max_waiting_time=max_waiting_time,
            directional_demand=directional_demand,
            occupancy=occupancy,
            vertiport_pmf=vertiport_pmf,
            vertiport_dict_inv=self.vertiport_dict_inv,
            flight_distance_matrix=self.flight_distance_matrix,
            fare=fare,
        )

        self.schedule = schedule = schedule
        self.pax_arrival_times = pax_arrival_times
