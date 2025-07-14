import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import googlemaps
import numpy as np
import pandas as pd
import pytz
from geopy.distance import distance
from geopy.point import Point
from tqdm import tqdm


class TravelTimeQuery:
    def __init__(self, coordiantes, vertiports, api_key):
        self.gmaps = googlemaps.Client(key=api_key)
        self.la_timezone = pytz.timezone("America/Los_Angeles")
        self.base_time = datetime(2025, 7, 15, 0, 0, 0, tzinfo=self.la_timezone)

        self.num_zones = len(vertiports)
        self.num_hours = 24
        self.coordinates = coordiantes

    def sample_travel_time(self, sample_size=10, n_cores=48):
        tnc_travel_time = np.zeros(
            (self.num_zones, self.num_zones, self.num_hours, sample_size)
        )
        first_mile_time = np.zeros((self.num_zones, self.num_hours, sample_size))
        last_mile_time = np.zeros((self.num_zones, self.num_hours, sample_size))
        first_last_mile_distances = np.zeros(
            (self.num_zones, self.num_hours, sample_size)
        )

        od_indices = list(range(1, len(self.coordinates)))

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = {
                executor.submit(self._process_od_idx, od_idx, sample_size): od_idx
                for od_idx in od_indices
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                od_idx, tnc_part, first_part, last_part, distance = future.result()
                tnc_travel_time[od_idx, 0] = tnc_part[0]
                tnc_travel_time[0, od_idx] = tnc_part[1]
                first_mile_time[od_idx] = first_part
                last_mile_time[od_idx] = last_part
                first_last_mile_distances[od_idx] = distance

        return (
            tnc_travel_time,
            first_mile_time,
            last_mile_time,
            first_last_mile_distances,
        )

    def _process_od_idx(self, od_idx, n):
        tnc_part = np.zeros((2, self.num_hours, n))  # [dtla→lax, lax→dtla]
        first_mile_part = np.zeros((self.num_hours, n))
        last_mile_part = np.zeros((self.num_hours, n))
        first_last_mile_distance = np.zeros((self.num_hours, n))

        radius = 2 if od_idx == 7 else 4

        # Sample `n` points once per OD index
        dtla_locations = self.generate_random_points(
            self.coordinates[od_idx], radius, n
        )

        for hour_offset in range(self.num_hours):
            query_time = self.base_time + timedelta(hours=hour_offset)
            unix_query_time = int(query_time.timestamp())
            hour = query_time.hour

            for i in range(n):
                dtla_location = dtla_locations[i]
                first_last_mile_distance[hour, i] = self.haversine_distance(
                    dtla_location, tuple(self.coordinates[od_idx])
                )

                travel_time_dtla_to_lax = self.get_travel_time(
                    dtla_location, tuple(self.coordinates[0]), unix_query_time
                )
                travel_time_lax_to_dtla = self.get_travel_time(
                    tuple(self.coordinates[0]), dtla_location, unix_query_time
                )
                travel_time_first_mile = self.get_travel_time(
                    tuple(self.coordinates[od_idx]), dtla_location, unix_query_time
                )
                travel_time_last_mile = self.get_travel_time(
                    dtla_location, tuple(self.coordinates[od_idx]), unix_query_time
                )

                tnc_part[0, hour, i] = self.convert_to_minutes(travel_time_dtla_to_lax)
                tnc_part[1, hour, i] = self.convert_to_minutes(travel_time_lax_to_dtla)
                first_mile_part[hour, i] = self.convert_to_minutes(
                    travel_time_first_mile
                )
                last_mile_part[hour, i] = self.convert_to_minutes(travel_time_last_mile)

        return (
            od_idx,
            tnc_part,
            first_mile_part,
            last_mile_part,
            first_last_mile_distance,
        )

    def get_travel_time(self, origin, destination, departure_time):
        result = self.gmaps.distance_matrix(
            origins=origin,
            destinations=destination,
            mode="driving",
            departure_time=departure_time,
        )
        if result["status"] == "OK":
            duration_in_traffic = result["rows"][0]["elements"][0].get(
                "duration_in_traffic", result["rows"][0]["elements"][0]["duration"]
            )
            return duration_in_traffic["text"]
        else:
            return f"Error: {result['status']}"

    @staticmethod
    def haversine_distance(coord1, coord2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 3956  # Radius of Earth in miles
        return c * r

    @staticmethod
    def generate_random_points(center_coords, radius, num_points):
        points = []
        for _ in range(num_points):
            # Random distance within the radius
            d = random.uniform(0, radius)
            # Random bearing in degrees
            bearing = random.uniform(0, 360)
            # Calculate the destination point
            point = distance(miles=d).destination(Point(center_coords), bearing)
            points.append((point.latitude, point.longitude))
        return points

    @staticmethod
    def convert_to_minutes(time_str):
        if "hour" in time_str:
            hours, mins = 0, 0
            parts = time_str.split(" ")
            for i in range(len(parts)):
                if parts[i] == "hour" or parts[i] == "hours":
                    hours = int(parts[i - 1])
                elif parts[i] == "min" or parts[i] == "mins":
                    mins = int(parts[i - 1])
            return hours * 60 + mins
        else:
            return int(time_str.split(" ")[0])
