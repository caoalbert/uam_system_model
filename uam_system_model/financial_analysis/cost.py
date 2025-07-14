import json

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

custom_colors = [
    "#c45161",
    "#e094a0",
    "#f2b6c0",
    "#f2dde1",
    "#cbc7d8",
    "#8db7d2",
    "#5e62a9",
    "#434279",
]
cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)
color_palette = np.array(
    [
        "#c45161",
        "#e094a0",
        "#f2b6c0",
        "#f2dde1",
        "#cbc7d8",
        "#8db7d2",
        "#5e62a9",
        "#434279",
    ]
)
matplotlib.rcParams.update({"legend.fontsize": 14, "legend.handlelength": 2})


class CostAnalyzer:
    def __init__(self, path_to_params):
        with open(path_to_params, "r") as file:
            self.params = json.load(file)
        self.capex = self.params["capex"]
        self.opex = self.params["opex"]

    def analyze(self, df):
        self.multiplier = 365 / df.shape[0]

        self.total_capex = self._compute_capex(df)
        self.total_opex = self._compute_opex(df)
        self.revenue = self._compute_revenue(df)

        return (
            round(self.total_capex, 2),
            round(self.total_opex, 2),
            round(self.revenue, 2),
            round(self.revenue - self.total_opex, 2),
        )

    def _compute_capex(self, df):
        fleet_size = int(df["fleet_size"][0])
        self.fleet_size = fleet_size
        for size in df["fleet_size"]:
            if fleet_size != size:
                raise Exception(
                    "Daily operation in the input file have different fleet sizes in rows"
                )
        self.fleet_aquisition_cost = fleet_size * self.capex["cost_per_vehicle"]

        required_pads = {}
        land_area = {}
        for row in df["pads_at_vertiport"]:
            for key, value in row.items():
                # Keep the max value seen so far for each key
                if key not in required_pads:
                    required_pads[key] = value
                else:
                    required_pads[key] = max(required_pads[key], value)

        for key, value in required_pads.items():
            land_area[key] = (
                self.capex["land"]["land_area_beta_0"]
                + value * self.capex["land"]["land_area_beta_1"]
            )
        land_cost = []
        for key, value in land_area.items():
            land_cost.append(
                value * self.capex["land"]["land_value_per_sqft"][key] * 10000
            )
            # print(f"Land cost for {key}: {land_cost[-1]}")
        land_acquisition_cost = sum(land_cost)
        self.land_acquisition_cost = land_acquisition_cost

        self.construction_cost = sum(self.capex["construction_cost"].values())

        return (
            self.fleet_aquisition_cost
            + self.construction_cost
            + self.land_acquisition_cost
        )

    def _compute_opex(self, df):

        self.energy_cost = (
            self.opex["energy_cost_per_kWh"] * df["energy_consumption_kWh"]
        ).sum()

        self.pilot_cost = (
            self.opex["pilot_cost_per_aircraft_hour"] * df["number_of_aircraft_hours"]
        ).sum()

        self.battery_replacement_cost = (
            self.opex["battery_replacement_cost_per_aircraft_hour"]
            * df["number_of_aircraft_hours"]
        ).sum()

        self.maintenance_cost = (
            self.opex["maintenance_cost_per_asm"] * df["TAM"] * 4
        ).sum()

        self.insurance_cost = (
            self.fleet_size
            * self.capex["cost_per_vehicle"]
            * self.opex["insurance_cost_factor"]
        )

        self.vertiport_operation_cost = self.opex[
            "vertiport_operation_cost_per_year"
        ] * len(df["pads_at_vertiport"][0])

        net_opex = (
            (
                self.energy_cost
                + self.pilot_cost
                + self.battery_replacement_cost
                + self.maintenance_cost
            )
            * self.multiplier
            + self.insurance_cost
            + self.vertiport_operation_cost
        )
        total_opex = net_opex / (1 - self.opex["administrative_cost_factor"])

        return total_opex

    def _compute_revenue(self, df):
        revenue = df["total_revenue"].sum()
        return revenue * self.multiplier

    def plot(self):
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        sns.barplot(
            x=["Fleet Aquisition Cost", "Construction Cost", "Land Acquisition Cost"],
            y=[
                self.fleet_aquisition_cost,
                self.construction_cost,
                self.land_acquisition_cost,
            ],
            ax=ax1,
            palette=color_palette[:3],
        )
        ax1.set_title("Total Capex, Opex, and Revenue")
        ax1.set_ylabel("Amount ($)")
        ax1.set_xlabel("Cost Type")

        sns.barplot(
            x=[
                "Energy Cost",
                "Pilot Cost",
                "Battery Replacement Cost",
                "Maintenance Cost",
                "Insurance Cost",
                "Vertiport Operation Cost",
                "Administrative Cost",
            ],
            y=[
                self.energy_cost * self.multiplier,
                self.pilot_cost * self.multiplier,
                self.battery_replacement_cost * self.multiplier,
                self.maintenance_cost * self.multiplier,
                self.insurance_cost,
                self.vertiport_operation_cost,
                self.total_opex * self.opex["administrative_cost_factor"],
            ],
            ax=ax2,
            palette=color_palette,
        )
        ax2.set_title("Opex Breakdown")
        ax2.set_ylabel("Amount ($)")
        ax2.set_xlabel("Cost Type")

        plt.tight_layout()
        plt.show()
