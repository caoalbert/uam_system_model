import pandas as pd

from .utils.schedule_utils import (generate_uam_schedule,
                                   generate_uam_schedule_v2)


class ScheduleGenerator:
    def __init__(self, path_schedule, path_seatcap):
        self.lax_flight = pd.read_csv(path_schedule)
        self.lax_flight = self.lax_flight[
            (self.lax_flight["USER_CLASS"] == "C")
            & (self.lax_flight["ON_YYYYMM"] <= 201912)
        ]
        self.lax_flight["T_OAG_S_DE"] = pd.to_datetime(self.lax_flight["T_OAG_S_DE"])
        self.lax_flight["T_OOOI_ARR"] = pd.to_datetime(self.lax_flight["T_OOOI_ARR"])

        self.seatcap = pd.read_csv(path_seatcap)
        seat_capacity = {
            "B752": 170,
            "B748": 300,
            "E75L": 60,
            "B77W": 300,
            "B739": 170,
            "A321": 170,
            "B737": 170,
            "B763": 300,
            "B744": 300,
            "B772": 300,
            "B738": 170,
            "B753": 170,
            "B789": 300,
            "B77L": 300,
            "A388": 300,
            "A332": 300,
            "CRJ2": 60,
            "B712": 60,
            "B788": 300,
            "CRJ7": 60,
            "E75S": 60,
            "A320": 170,
            "A319": 170,
            "A359": 300,
            "E55P": 60,
            "B764": 300,
            "E190": 60,
            "A333": 300,
            "C680": 0,
            "A21N": 170,
            "F2TH": 0,
            "A343": 300,
            "C208": 0,
            "B350": 300,
            "E50P": 0,
            "PC12": 0,
            "B78X": 300,
            "C56X": 0,
            "LJ60": 0,
            "CL35": 0,
            "A20N": 170,
            "B732": 170,
        }

        seat_capacity = pd.DataFrame.from_dict(
            seat_capacity, orient="index"
        ).reset_index()
        seat_capacity.columns = ["ETMS_EQPT", "capacity"]

        self.lax_flight["TAILNO"] = self.lax_flight["TAILNO"].str.strip()

        self.lax_flight = self.lax_flight.merge(
            self.seatcap[["TAIL_NUMBER", "NUMBER_OF_SEATS"]],
            left_on="TAILNO",
            right_on="TAIL_NUMBER",
            how="left",
        )

        self.lax_flight = self.lax_flight.merge(
            seat_capacity, on="ETMS_EQPT", how="left"
        )
        self.lax_flight["capacity"] = self.lax_flight["NUMBER_OF_SEATS"].fillna(
            self.lax_flight["capacity"]
        )
        self.lax_flight = self.lax_flight.dropna(subset=["capacity"])

        arr_capacity = self.lax_flight[(self.lax_flight["ARR_LOCID"] == " LAX")][
            "capacity"
        ].sum()
        dep_capacity = self.lax_flight[(self.lax_flight["DEP_LOCID"] == " LAX")][
            "capacity"
        ].sum()

        self.yearly_capacity = (arr_capacity, dep_capacity)

    def get_one_day(
        self,
        month,
        day,
        auto_regressive_alpha,
        max_waiting_time,
        directional_demand,
        occupancy,
        vertiport_pmf,
        vertiport_dict_inv,
        flight_distance_matrix,
        fare,
    ):
        month = month + 201900

        lax_flight_arr = self.lax_flight[
            (self.lax_flight["ON_YYYYMM"] == month)
            & (self.lax_flight["ON_DAY"] == day)
            & (self.lax_flight["ARR_LOCID"] == " LAX")
        ]
        lax_flight_arr["time"] = (
            lax_flight_arr["T_OOOI_ARR"].dt.hour * 60
            + lax_flight_arr["T_OOOI_ARR"].dt.minute
        )
        lax_flight_arr = lax_flight_arr.sort_values("time").reset_index(drop=True)
        lax_flight_dep = self.lax_flight[
            (self.lax_flight["ON_YYYYMM"] == month)
            & (self.lax_flight["ON_DAY"] == day)
            & (self.lax_flight["DEP_LOCID"] == " LAX")
        ]
        lax_flight_dep["time"] = (
            lax_flight_dep["T_OAG_S_DE"].dt.hour * 60
            + lax_flight_dep["T_OAG_S_DE"].dt.minute
        )
        lax_flight_dep = lax_flight_dep.sort_values("time").reset_index(drop=True)

        self.lax_flight_arr = lax_flight_arr
        self.lax_flight_dep = lax_flight_dep

        if len(vertiport_pmf.shape) == 1:
            if directional_demand is None:
                raise ValueError(
                    "Directional demand must be specified when vertiport_pmf is 1D."
                )
            schedule, pax_arrival_times = generate_uam_schedule(
                lax_flight_arr,
                lax_flight_dep,
                self.yearly_capacity,
                auto_regressive_alpha,
                directional_demand,
                vertiport_pmf,
                occupancy,
                max_waiting_time,
                vertiport_dict_inv,
                flight_distance_matrix,
                fare,
            )
        elif len(vertiport_pmf.shape) == 3:
            schedule, pax_arrival_times = generate_uam_schedule_v2(
                lax_flight_arr,
                lax_flight_dep,
                auto_regressive_alpha,
                vertiport_pmf,
                occupancy,
                max_waiting_time,
                vertiport_dict_inv,
                flight_distance_matrix,
                fare,
            )
        else:
            raise ValueError(
                "vertiport_pmf must be a 1D or 3D array, got shape: {}".format(
                    vertiport_pmf.shape
                )
            )

        return schedule, pax_arrival_times
