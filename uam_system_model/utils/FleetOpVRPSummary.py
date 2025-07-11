from uam_system_model.FleetOpVRP import FleetOpVRP
import numpy as np
import pandas as pd


class FleetOpVRPSummary(FleetOpVRP):
    def __init__(self, StarNetwork, policy, non_linear_battery_charging=True):

        super().__init__(StarNetwork)
        if non_linear_battery_charging:
            self.cs = np.cumsum(
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
        else:
            self.cs = np.cumsum(np.array([0] + [4.3085625] * 32))

        self.policy = policy
        self._append_column()

    def get_summary_statistics(self):
        fleet_size = self.policy["cluster_id"].nunique()

        parking_pads = self._infer_num_pads()
        total_pads = parking_pads["num_pads"].sum()
        num_pads_at_vertiport = {
            self.network.vertiport_dict_inv[i]: parking_pads[
                parking_pads["vertiport_idx"] == i
            ]["num_pads"].values[0]
            for i in range(len(self.network.vertiport_dict))
        }

        max_takeoff_per_hour = self._infer_num_tlof()
        max_takeoff_per_hour_at_vertiport = {
            self.network.vertiport_dict_inv[i]: max_takeoff_per_hour[
                max_takeoff_per_hour["location"] == i
            ]["count"].values[0]
            for i in range(len(self.network.vertiport_dict))
        }

        repo_flight_duration = self.policy["repo_flight_duration"].sum() / 60
        revenue_flight_duration = self.policy["duration"].sum() / 60
        aircraft_hours = repo_flight_duration + revenue_flight_duration

        repositioning_flights = self.policy["is_repo"].sum()
        total_number_of_flights = (
            repositioning_flights + self.policy["tour_length"].sum()
        )
        energy_consumption_counted = (
            self.policy["q"] - self.policy["p"]
        ).values.sum() + (self.policy["q2"] - self.policy["p2"]).values.sum()
        last_entries = self.policy.groupby("cluster_id").tail(1)
        energy_consumption_not_counted = (100 - last_entries["q2"].values).sum()
        energy_consumption = (
            energy_consumption_counted + energy_consumption_not_counted / 100 * 160
        )

        repo_aircraft_miles = self.policy["repo_flight_miles"].sum()
        revenue_aircraft_miles = 0
        for idx, row in self.policy.iterrows():
            sequence = row["tour_sequence"].split("-")
            for i in range(len(sequence) - 1):
                origin = self.network.vertiport_dict[sequence[i]]
                destination = self.network.vertiport_dict[sequence[i + 1]]
                revenue_aircraft_miles += self.network.flight_distance_matrix[origin][
                    destination
                ]

        total_aircraft_miles = repo_aircraft_miles + revenue_aircraft_miles

        load_factor = self.network.schedule["num_pax"].mean() / 4

        summary = {
            "fleet_size": int(fleet_size),
            "total_pads": total_pads,
            "pads_at_vertiport": num_pads_at_vertiport,
            "max_takeoff_per_hour": max_takeoff_per_hour_at_vertiport,
            "number_of_aircraft_hours": round(aircraft_hours, 2),
            "number_of_flights": int(total_number_of_flights),
            "number_of_revenue_flights": int(self.policy["tour_length"].sum()),
            "number_of_repositioning_flights": int(repositioning_flights),
            "percentage_repositioning_flights": round(
                repositioning_flights / total_number_of_flights, 4
            ),
            "energy_consumption_kWh": round(energy_consumption, 2),
            "TAM": round(total_aircraft_miles, 2),
            "RAM": round(revenue_aircraft_miles, 2),
            "total_revenue": self.policy['tour_revenue'].sum(),
            "average_load_factor": round(load_factor, 4),
        }

        return summary

    def _append_column(self):
        self.policy["arrival_time"] = self.policy["time"] + self.policy["duration"]
        self.policy["next_origin"] = self.policy.groupby("cluster_id")["origin"].shift(
            -1
        )
        self.policy["next_flight_start_time"] = self.policy.groupby("cluster_id")[
            "time"
        ].shift(-1)
        self.policy["next_origin"] = self.policy["next_origin"].fillna(999)
        self.policy["repo_flight_duration"] = self.policy.apply(
            lambda row: self.network.flight_time[
                int(row["destination"]), int(row["next_origin"])
            ]
            * 5
            if row["next_origin"] != 999
            else 0,
            axis=1,
        )
        self.policy["repo_flight_miles"] = self.policy.apply(
            lambda row: self.network.flight_distance_matrix[
                int(row["destination"]), int(row["next_origin"])
            ]
            if row["next_origin"] != 999
            else 0,
            axis=1,
        )
        self.policy["is_repo"] = self.policy.apply(
            lambda row: 1
            if row["destination"] != row["next_origin"] and row["next_origin"] != 999
            else 0,
            axis=1,
        )
        self.policy["charge_time_before_repo"] = self.policy.apply(
            lambda row: self._calc_charging_time(row["p"], row["q"]), axis=1
        )
        # self.policy['done_charge_time_before_repo'] = self.policy["time"] + self.policy["duration"] + self.policy['charge_time_before_repo']
        self.policy["charge_time_after_repo"] = self.policy.apply(
            lambda row: self._calc_charging_time(row["p2"], row["q2"]), axis=1
        )
        # self.policy['done_charge_time_after_repo'] = self.policy["done_charge_time_before_repo"] + self.policy['charge_time_after_repo']
        self.policy["reposition_start_time"] = (
            self.policy["time"]
            + self.policy["duration"]
            + self.policy["charge_time_before_repo"]
        )
        self.policy["reposition_end_time"] = (
            self.policy["reposition_start_time"] + self.policy["repo_flight_duration"]
        )

        self.policy["reposition_hour"] = self.policy["reposition_start_time"] // 60
        self.policy["revenue_flight_hour"] = self.policy["time"] // 60

    def _calc_charging_time(self, p, q):
        int_p = int(np.round((p - 20) / 2.5, 1))
        int_q = int(np.round((q - 20) / 2.5, 1))
        charge_time = self.cs[int_q] - self.cs[int_p]
        return charge_time

    def _infer_num_pads(self):
        # two intervals on ground
        # 1. [arrival_time, reposition_start_time]
        # 2. [reposition_end_time, next_flight_start_time]
        intervals = pd.DataFrame(
            columns=["start", "end", "location"]
        )  # DataFrame to store intervals

        for index, row in self.policy.iterrows():
            interval_1 = {
                "start": row["arrival_time"],
                "end": row["reposition_start_time"]
                if pd.notna(row["reposition_start_time"])
                else 10000,
                "location": row["destination"],
            }
            interval_2 = {
                "start": row["reposition_end_time"],
                "end": row["next_flight_start_time"]
                if pd.notna(row["next_flight_start_time"])
                else 10000,
                "location": row["next_origin"]
                if row["next_origin"] != 999
                else row["destination"],
            }
            intervals = intervals.append(interval_1, ignore_index=True)
            intervals = intervals.append(interval_2, ignore_index=True)

        results = []
        for loc, group in intervals.groupby("location"):
            events = []
            for _, row in group.iterrows():
                start = row["start"]
                end = row["end"]
                events.append((start, +1))
                events.append((end, -1))

            events.sort(key=lambda x: (x[0], -x[1]))

            current = 0
            max_overlap = 0
            max_times = []

            for time, change in events:
                current += change
                if current > max_overlap:
                    max_overlap = current
                    max_times = [time]
                elif current == max_overlap:
                    max_times.append(time)

            results.append(
                {
                    "vertiport_idx": loc,
                    "num_pads": max_overlap,
                    "at_times": sorted(set(max_times)),
                }
            )

        return pd.DataFrame(results)

    def _infer_num_tlof(self):
        df = pd.DataFrame(columns=["hour", "location"])
        for index, row in self.policy.iterrows():
            if row["is_repo"]:
                df = df.append(
                    {"hour": row["reposition_hour"], "location": row["next_origin"]},
                    ignore_index=True,
                )
            df = df.append(
                {"hour": row["revenue_flight_hour"], "location": row["origin"]},
                ignore_index=True,
            )
            sequence = row["tour_sequence"].split("-")
            if len(sequence) < 2:
                continue
            for i in range(1, len(sequence) - 1):
                df = df.append(
                    {"hour": row["revenue_flight_hour"], "location": row["origin"]},
                    ignore_index=True,
                )

        ops_per_hour = df.groupby(["hour", "location"]).size().reset_index(name="count")
        max_dep_per_hour = ops_per_hour.groupby("location")["count"].max().reset_index()

        return max_dep_per_hour
