import numpy as np
import pandas as pd

from .utils.schedule_utils import auto_regressive_poisson, build_schedules


class ScheduleGenerator:
    def __init__(self):
        pass

    def get_one_day(
        self,
        auto_regressive_alpha,
        max_waiting_time,
        occupancy,
        arrival_rate_by_vertiport,
        vertiport_dict,
        flight_distance_matrix,
        fare,
    ):
        list_of_ods = arrival_rate_by_vertiport["od"].unique()

        schedule = pd.DataFrame(columns=["schedule", "od", "num_pax", "revenue"])
        paxArrivalDf = pd.DataFrame(
            columns=[
                "passenger_arrival_time_s",
                "origin_vertiport_id",
                "destination_vertiport_id",
            ]
        )

        for od in list_of_ods:
            od_data = arrival_rate_by_vertiport[arrival_rate_by_vertiport["od"] == od]
            realized_demand = auto_regressive_poisson(
                rate=od_data["total_trips"].values / 365,
                alpha=auto_regressive_alpha,
            )
            arrival_time = []
            for hour in range(24):
                for _ in range(realized_demand[hour]):
                    arrival_time.append(hour * 60 + np.random.randint(0, 60))

            departure_time, num_pax = build_schedules(
                arrival_time, max_waiting_time=max_waiting_time, occupancy=occupancy
            )

            schedule = pd.concat(
                [
                    schedule,
                    pd.DataFrame(
                        {
                            "schedule": departure_time,
                            "od": od,
                            "num_pax": num_pax,
                            "revenue": num_pax
                            * flight_distance_matrix[
                                vertiport_dict[od.split("_")[0]],
                                vertiport_dict[od.split("_")[1]],
                            ]
                            * fare,  # assume flat fare of $50
                        }
                    ),
                ],
                ignore_index=True,
            )

            arrival_time = np.array(arrival_time)
            paxArrivalDf = pd.concat(
                [
                    paxArrivalDf,
                    pd.DataFrame(
                        {
                            "passenger_arrival_time_s": arrival_time * 60,
                            "origin_vertiport_id": vertiport_dict[od.split("_")[0]],
                            "destination_vertiport_id": vertiport_dict[
                                od.split("_")[1]
                            ],
                        }
                    ),
                ],
                ignore_index=True,
            )

        schedule = schedule.sort_values(by=["schedule", "od"]).reset_index(drop=True)

        paxArrivalDf = paxArrivalDf.reset_index(drop=False)
        paxArrivalDf = paxArrivalDf.rename(columns={"index": "passenger_id"})

        return schedule, paxArrivalDf
