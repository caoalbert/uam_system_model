import warnings

import numpy as np
import pandas as pd
from scipy.stats import skewnorm

warnings.filterwarnings("ignore")

# v2 is for knowing the total passenger demand from each spoke
# v1 is for assuming a fixed number of daily passengers from each spoke


def generate_uam_schedule_v2(
    lax_flight_arr,
    lax_flight_dep,
    auto_regressive_alpha,
    vertiport_pmf,
    occupancy,
    max_waiting_time,
    vertiport_dict_inv,
    flight_distance_matrix,
    fare,
):
    lax_flight_arr = get_autoregressive_pax_count_v2(
        lax_flight_arr, vertiport_pmf, "LAX-Spoke", auto_regressive_alpha
    )
    lax_flight_dep = get_autoregressive_pax_count_v2(
        lax_flight_dep, vertiport_pmf, "Spoke-LAX", auto_regressive_alpha
    )

    origin_name = []
    destination_name = []

    # Generate passenger arrival times
    lax_dtla = np.empty((1,))
    for idx, row in lax_flight_arr.iterrows():
        num_pax = row["capacity"]
        delta_t = skewnorm.rvs(3, loc=31, scale=np.sqrt(2 * 1.5**2), size=num_pax)
        delta_t += row["time"]
        lax_dtla = np.concatenate([lax_dtla, delta_t])
        origin_name += [0] * num_pax
        destination_name += [row["destination"]] * num_pax
    lax_dtla[lax_dtla >= 1440] -= 1440
    lax_dtla = lax_dtla[1:]

    dtla_lax = np.empty((1,))
    for idx, row in lax_flight_dep.iterrows():
        num_pax = row["capacity"]
        delta_t = skewnorm.rvs(3, loc=93, scale=40, size=num_pax)
        delta_t += np.random.normal(loc=10, scale=5 / 3, size=num_pax)
        delta_t = row["time"] - delta_t
        dtla_lax = np.concatenate([dtla_lax, delta_t])
        origin_name += [row["origin"]] * num_pax
        destination_name += [0] * num_pax
    dtla_lax[dtla_lax < 0] += 1440
    dtla_lax = dtla_lax[1:]

    pax_arrival_times = np.concatenate(
        [np.round(lax_dtla * 60), np.round(dtla_lax * 60)], axis=0
    )
    passenger_id = np.arange(0, len(pax_arrival_times))

    paxArrivalDf = (
        pd.DataFrame(
            {
                "passenger_id": passenger_id,
                "passenger_arrival_time_s": pax_arrival_times,
                "origin_vertiport_id": origin_name,
                "destination_vertiport_id": destination_name,
            }
        )
        .sort_values(by="passenger_arrival_time_s")
        .reset_index(drop=True)
    )

    final_schedule = pd.DataFrame(columns=["schedule", "od", "num_pax", "revenue"])

    for i in range(1, vertiport_pmf.shape[1]):
        # CBD to LAX vertiport
        dtla_lax_sche, dtla_lax_num_pax_flight = build_schedules(
            paxArrivalDf[paxArrivalDf["origin_vertiport_id"] == i][
                "passenger_arrival_time_s"
            ].values
            / 60,
            max_waiting_time,
            occupancy,
        )
        final_schedule = pd.concat(
            [
                final_schedule,
                pd.DataFrame(
                    {
                        "schedule": dtla_lax_sche,
                        "od": f"{vertiport_dict_inv[i]}_LAX",
                        "num_pax": dtla_lax_num_pax_flight,
                        "revenue": dtla_lax_num_pax_flight
                        * flight_distance_matrix[i, 0]
                        * fare,
                    }
                ),
            ],
            ignore_index=True,
        )

        # LAX TO CBD vertiport
        lax_dtla_sche, lax_dtla_num_pax_flight = build_schedules(
            paxArrivalDf[paxArrivalDf["destination_vertiport_id"] == i][
                "passenger_arrival_time_s"
            ].values
            / 60,
            max_waiting_time,
            occupancy,
        )
        final_schedule = pd.concat(
            [
                final_schedule,
                pd.DataFrame(
                    {
                        "schedule": lax_dtla_sche,
                        "od": f"LAX_{vertiport_dict_inv[i]}",
                        "num_pax": lax_dtla_num_pax_flight,
                        "revenue": lax_dtla_num_pax_flight
                        * flight_distance_matrix[0, i]
                        * fare,
                    }
                ),
            ],
            ignore_index=True,
        )

    final_schedule = final_schedule.sort_values(by=["schedule", "od"]).reset_index(
        drop=True
    )

    return final_schedule, paxArrivalDf


def generate_uam_schedule(
    hub_flight_arr,
    hub_flight_dep,
    total_hub_seats,
    auto_regressive_alpha,
    directional_demand,
    vertiport_pmf,
    occupancy,
    max_waiting_time,
    vertiport_dict_inv,
    hub_name,
    flight_distance_matrix,
    fare,
):
    yearly_arr_capacity, yearly_dep_capacity = total_hub_seats
    # Calculate pax demand per scheduled flight
    hub_flight_arr["capacity"] = (
        hub_flight_arr["capacity"] / yearly_arr_capacity * directional_demand * 365
    )
    hub_flight_dep["capacity"] = (
        hub_flight_dep["capacity"] / yearly_dep_capacity * directional_demand * 365
    )

    hub_flight_arr = get_autoregressive_pax_count(hub_flight_arr, auto_regressive_alpha)
    hub_flight_dep = get_autoregressive_pax_count(hub_flight_dep, auto_regressive_alpha)
    # Generate passenger arrival times
    lax_dtla = np.empty((1,))
    for i in range(hub_flight_arr.shape[0]):
        num_pax = hub_flight_arr.iloc[i, 1]
        delta_t = skewnorm.rvs(3, loc=31, scale=np.sqrt(2 * 1.5**2), size=num_pax)
        delta_t += hub_flight_arr.iloc[i, 0]
        lax_dtla = np.concatenate([lax_dtla, delta_t])
    lax_dtla[lax_dtla >= 1440] -= 1440
    lax_dtla = np.sort(lax_dtla)

    dtla_lax = np.empty((1,))
    for i in range(hub_flight_dep.shape[0]):
        num_pax = hub_flight_dep.iloc[i, 1]
        delta_t = skewnorm.rvs(3, loc=93, scale=40, size=num_pax)
        delta_t += np.random.normal(loc=10, scale=5 / 3, size=num_pax)
        delta_t = hub_flight_dep.iloc[i, 0] - delta_t
        dtla_lax = np.concatenate([dtla_lax, delta_t])
    dtla_lax[dtla_lax < 0] += 1440
    dtla_lax = np.sort(dtla_lax)

    # Assign passenger to particular network
    pax_arrival_times = np.concatenate(
        [np.round(np.sort(dtla_lax) * 60), np.round(np.sort(lax_dtla) * 60)], axis=0
    )
    passenger_id = np.arange(0, len(pax_arrival_times))

    destination = inverse_cdf(vertiport_pmf.cumsum(), lax_dtla.shape[0])
    origin = inverse_cdf(vertiport_pmf.cumsum(), dtla_lax.shape[0])

    origin_name = np.concatenate(
        [
            np.array([vertiport_dict_inv[int(i)] for i in origin]),
            np.repeat(hub_name, len(lax_dtla)),
        ],
        axis=0,
    )
    destination_name = np.concatenate(
        [
            np.repeat(hub_name, len(dtla_lax)),
            np.array([vertiport_dict_inv[int(i)] for i in destination]),
        ],
        axis=0,
    )

    paxArrivalDf = pd.DataFrame(
        {
            "passenger_id": passenger_id,
            "passenger_arrival_time_s": pax_arrival_times,
            "origin_vertiport_id": origin_name,
            "destination_vertiport_id": destination_name,
        }
    )

    final_schedule = pd.DataFrame(columns=["schedule", "od", "num_pax", "revenue"])

    for i in range(1, len(vertiport_pmf)):
        # CBD to LAX vertiport
        dtla_lax_sche, dtla_lax_num_pax_flight = build_schedules(
            dtla_lax[origin == i], max_waiting_time, occupancy
        )
        final_schedule = pd.concat(
            [
                final_schedule,
                pd.DataFrame(
                    {
                        "schedule": dtla_lax_sche,
                        "od": f"{vertiport_dict_inv[i]}_{hub_name}",
                        "num_pax": dtla_lax_num_pax_flight,
                        "revenue": dtla_lax_num_pax_flight
                        * flight_distance_matrix[i, 0]
                        * fare,
                    }
                ),
            ],
            ignore_index=True,
        )

        # LAX TO CBD vertiport
        lax_dtla_sche, lax_dtla_num_pax_flight = build_schedules(
            lax_dtla[destination == i], max_waiting_time, occupancy
        )
        final_schedule = pd.concat(
            [
                final_schedule,
                pd.DataFrame(
                    {
                        "schedule": lax_dtla_sche,
                        "od": f"{hub_name}_{vertiport_dict_inv[i]}",
                        "num_pax": lax_dtla_num_pax_flight,
                        "revenue": lax_dtla_num_pax_flight
                        * flight_distance_matrix[0, i]
                        * fare,
                    }
                ),
            ],
            ignore_index=True,
        )

    final_schedule = final_schedule.sort_values(by=["schedule", "od"]).reset_index(
        drop=True
    )

    return final_schedule, paxArrivalDf


def build_schedules(queue, max_waiting_time, occupancy):
    queue = np.asarray(queue)
    if queue.size == 0:
        return np.array([]), np.array([])

    queue = np.sort(queue)
    departure_time_aircraft = []
    num_pax_per_flight = []
    cnt = 0
    first_pax = queue[0]
    for i in range(len(queue)):
        cnt += 1
        if (queue[i] - first_pax) > max_waiting_time:
            departure_time_aircraft.append(first_pax + max_waiting_time)
            num_pax_per_flight.append(cnt - 1)
            cnt = 1
            first_pax = queue[i]

        elif cnt == occupancy:
            departure_time_aircraft.append(queue[i])
            num_pax_per_flight.append(occupancy)
            cnt = 0
            if i + 1 < len(queue):
                first_pax = queue[i + 1]

    # Add remaining passengers if any
    if len(departure_time_aircraft) == 0 or queue[-1] > departure_time_aircraft[-1]:
        departure_time_aircraft.append(queue[-1])
        num_pax_per_flight.append(cnt)

    return np.array(departure_time_aircraft), np.array(num_pax_per_flight)


def get_autoregressive_pax_count(arr_merged, auto_regressive_alpha):
    # Calculate hourly rate
    arr_merged_v2 = arr_merged.copy()
    arr_merged_v2["time_interval"] = arr_merged_v2["time"] // 60
    arr_merged_v2_grouped = arr_merged_v2.groupby("time_interval").sum("capacity")
    arr_merged_v2_grouped = (
        arr_merged_v2_grouped.merge(
            pd.DataFrame({"time_interval": np.arange(0, 24, 1), "demand": 0}),
            on="time_interval",
            how="outer",
        )
        .sort_values("time_interval")
        .reset_index(drop=True)
        .fillna(0)
    )

    # Calculate flight pax density over hourly rate
    arr_merged_v2_merged = arr_merged_v2.merge(
        arr_merged_v2_grouped, on="time_interval"
    )
    arr_merged_v2_merged["density"] = (
        arr_merged_v2_merged["capacity_x"] / arr_merged_v2_merged["capacity_y"]
    )
    arr_merged_v2_merged["cum_density"] = arr_merged_v2_merged.groupby("time_interval")[
        "density"
    ].cumsum()

    # Generate one realization of hourly rate
    realized_rate = auto_regressive_poisson(
        arr_merged_v2_grouped["capacity"].values, alpha=auto_regressive_alpha
    )

    # Assign hourly rate to each flight by inverse cdf
    assigned_pax = np.empty(shape=(1,), dtype=int)
    for i in range(24):
        check = arr_merged_v2_merged[arr_merged_v2_merged["time_interval"] == i]
        if check.shape[0] == 0:
            continue
        x = inverse_cdf(check["cum_density"].values, num_samples=realized_rate[i])
        assigned_pax_i = np.histogram(x, np.arange(0, check.shape[0] + 1))[0]
        assigned_pax = np.concatenate([assigned_pax, assigned_pax_i])

    assigned_pax = assigned_pax[1:]
    output = pd.DataFrame({"time": arr_merged["time"], "capacity": assigned_pax})

    return output


def auto_regressive_poisson(rate, alpha):
    lambda_0_0 = rate[0]
    x_0 = np.random.poisson(lambda_0_0)
    output = [x_0]
    for i in range(1, len(rate)):
        lambda_1_0 = rate[i]

        if lambda_1_0 == 0:
            x_0 = 0
        else:
            if lambda_0_0 == 0:
                lambda_1 = lambda_1_0
            else:
                lambda_1 = (
                    lambda_1_0 + (x_0 - lambda_0_0) * lambda_1_0 / lambda_0_0 * alpha
                )
            x_0 = np.random.poisson(lambda_1)

        lambda_0_0 = lambda_1_0
        output.append(x_0)

    return np.array(output)


def inverse_cdf(cumulative_pmf, num_samples):
    samples = []
    for _ in range(num_samples):
        u = np.random.uniform(0, 1)
        sample_index = u <= cumulative_pmf
        samples.append(np.argmax(sample_index))

    samples = np.array(samples)
    return samples


def get_autoregressive_pax_count_v2(
    lax_flight, vertiport_pmf, direction, auto_regressive_alpha
):
    if direction == "LAX-Spoke":
        alpha_index = 0
    elif direction == "Spoke-LAX":
        alpha_index = 1

    lax_flight["time_interval"] = lax_flight["time"] // 60
    lax_flight_grouped = lax_flight.groupby("time_interval").sum("capacity")
    lax_flight_grouped = (
        lax_flight_grouped.merge(
            pd.DataFrame({"time_interval": np.arange(0, 24, 1), "demand": 0}),
            on="time_interval",
            how="outer",
        )
        .sort_values("time_interval")
        .reset_index(drop=True)
        .fillna(0)
    )

    # Calculate flight pax density over hourly rate
    lax_flight_merged = lax_flight.merge(lax_flight_grouped, on="time_interval")
    lax_flight_merged["density"] = (
        lax_flight_merged["capacity_x"] / lax_flight_merged["capacity_y"]
    )
    lax_flight_merged["cum_density"] = lax_flight_merged.groupby("time_interval")[
        "density"
    ].cumsum()

    # Generate one realization of hourly rate
    lax_dtla_hourly_uam_demand = (
        vertiport_pmf[alpha_index] * lax_flight_grouped["capacity"].values
    )

    output = pd.DataFrame(columns=["time", "capacity", "origin", "destination"])

    assigned_pax = np.empty(shape=(1,), dtype=int)
    for od in range(1, lax_dtla_hourly_uam_demand.shape[0]):
        realized_rate = auto_regressive_poisson(
            lax_dtla_hourly_uam_demand[od], auto_regressive_alpha
        )

        assigned_pax = np.empty(shape=(1,), dtype=int)
        for i in range(24):
            check = lax_flight_merged[lax_flight_merged["time_interval"] == i]
            if check.shape[0] == 0:
                continue
            x = inverse_cdf(check["cum_density"].values, num_samples=realized_rate[i])
            assigned_pax_i = np.histogram(x, np.arange(0, check.shape[0] + 1))[0]
            assigned_pax = np.concatenate([assigned_pax, assigned_pax_i])

        assigned_pax = assigned_pax[1:]
        if alpha_index == 0:
            output = pd.concat(
                [
                    output,
                    pd.DataFrame(
                        {
                            "time": lax_flight["time"],
                            "capacity": assigned_pax,
                            "origin": 0,
                            "destination": od,
                        }
                    ),
                ],
                ignore_index=True,
            )
        elif alpha_index == 1:
            output = pd.concat(
                [
                    output,
                    pd.DataFrame(
                        {
                            "time": lax_flight["time"],
                            "capacity": assigned_pax,
                            "origin": od,
                            "destination": 0,
                        }
                    ),
                ],
                ignore_index=True,
            )
    return output
