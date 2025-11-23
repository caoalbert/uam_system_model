import json

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf
import pandas as pd
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
color = ["#93003a", "#00429d", "#93c4d2", "#6ebf7c"]
cmap = LinearSegmentedColormap.from_list("custom_cmap", custom_colors)
matplotlib.rcParams.update(
    {
        "legend.fontsize": 14,
        "legend.handlelength": 2,
        "xtick.labelsize": 18,
        "ytick.labelsize": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
    }
)


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

        balance = {}
        balance['RASM'] = self.revenue / 365 / df['TAM'][0] / 4
        balance['CASM'] = {}
        balance['CASM']['TOTAL'] = self.total_opex / 365 / df['TAM'][0] / 4
        balance['CASM']['ENERGY'] = (self.energy_cost * self.multiplier) / 365 / df['TAM'][0] / 4
        balance['CASM']['PILOT'] = (self.pilot_cost * self.multiplier) / 365 / df['TAM'][0] / 4
        balance['CASM']['BATTERY_REPLACEMENT'] = (self.battery_replacement_cost * self.multiplier) / 365 / df['TAM'][0] / 4
        balance['CASM']['MAINTENANCE'] = (self.maintenance_cost * self.multiplier) / 365 / df['TAM'][0] / 4
        balance['CASM']['INSURANCE'] = self.insurance_cost / 365 / df['TAM'][0] / 4
        balance['CASM']['VERTIPORT_OPERATION'] = self.vertiport_operation_cost / 365 / df['TAM'][0] / 4
        balance['CASM']['ADMINISTRATION'] = (self.total_opex * self.opex["administrative_cost_factor"]) / 365 / df['TAM'][0] / 4
        self.balance = balance
        print("RASM ......................... : ${:.3f}".format(balance['RASM']))
        print("CASM ......................... : ${:.3f}".format(balance['CASM']['TOTAL']))
        for key, value in balance['CASM'].items():
            if key != 'TOTAL':
                print(f"  - {key:<24} : ${value:.3f}")

        return (
            round(self.total_capex, 2),
            round(self.total_opex, 2),
            round(self.revenue, 2),
            round(self.revenue - self.total_opex, 2),
        )

    def projection(self, years=15, discount_rate=0.08):
        rev_by_year = np.full(years, self.revenue)
        opex_by_year = np.full(years, self.total_opex)
        net_cash_flow = np.concatenate(
            ([-self.total_capex], rev_by_year - opex_by_year)
        )

        discount_factors = 1 / (1 + discount_rate) ** np.arange(years + 1)
        discounted_cash_flows = net_cash_flow * discount_factors

        ROI = discounted_cash_flows.sum() / self.total_capex
        NPV = discounted_cash_flows.sum()
        IRR = npf.irr(net_cash_flow)
        cum_disc = np.cumsum(discounted_cash_flows)
        payback_year = next((t for t, v in enumerate(cum_disc) if v >= 0), None)

        fig, ax = self._plot_projection(
            discount_rate,
            years,
            net_cash_flow,
            discounted_cash_flows,
            np.cumsum(net_cash_flow),
            cum_disc,
        )

        print(f"ROI (undiscounted) ............ : {ROI:.2f}")
        print(f"NPV @ {discount_rate:.0%} ............. : ${NPV:,.0f}")
        print(f"IRR ........................... : {IRR:.2%}")
        print(
            f"Discounted Payback ............ : " f"{payback_year} years"
            if payback_year is not None
            else "Not recovered"
        )

        return fig, ax, ROI, NPV, IRR, payback_year

    def _plot_projection(
        self,
        discount_rate,
        year,
        cash_flow,
        discounted_cashflow,
        cumulative_cash_flow,
        cumulative_discounted_cashflow,
    ):

        df = pd.DataFrame(
            {
                "Year": np.arange(year + 1),
                "Cash Flow": cash_flow,
                "Discounted CF": discounted_cashflow,
                "Cumulative CF": cumulative_cash_flow,
                "Cumulative Discounted CF": cumulative_discounted_cashflow,
            }
        )

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        ax.bar(
            df["Year"],
            df["Cash Flow"] / 1e6,
            color=color[1],
            alpha=0.7,
            label="Annual Operating Profit",
        )
        ax.plot(
            df["Year"],
            df["Cumulative CF"] / 1e6,
            "k--",
            lw=2,
            label="Cumulative Cash Flow",
        )
        ax.plot(
            df["Year"],
            df["Cumulative Discounted CF"] / 1e6,
            "r",
            lw=2,
            label=f"Cumulative Discounted CF ({discount_rate:.0%})",
        )

        ax.set(
            xlim=(-0.5, year),
            xticks=np.arange(0, year + 1, 2),
            xlabel="Year",
            ylabel="Million ($)",
        )
        ax.legend()
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.25, linestyle="--", which="both")
        plt.tight_layout()

        return fig, ax

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
        fig, ax = plt.subplots(ncols=2, figsize=(14, 4), dpi=300)
        sns.barplot(
            x=["Fleet Aquisition Cost", "Construction Cost", "Land Acquisition Cost"],
            y=[
                self.fleet_aquisition_cost / 1e6,
                self.construction_cost / 1e6,
                self.land_acquisition_cost / 1e6,
            ],
            ax=ax[0],
            palette=color[:3],
        )
        ax[0].set_title("CAPEX", fontsize=24)

        sns.barplot(
            x=[
                "Energy",
                "Pilot",
                "Battery \n Replacement",
                "Maintenance",
                "Insurance",
                "Vertiport \n Operation",
                "Administration",
            ],
            y=[
                self.energy_cost * self.multiplier / 1e6,
                self.pilot_cost * self.multiplier / 1e6,
                self.battery_replacement_cost * self.multiplier / 1e6,
                self.maintenance_cost * self.multiplier / 1e6,
                self.insurance_cost / 1e6,
                self.vertiport_operation_cost / 1e6,
                self.total_opex * self.opex["administrative_cost_factor"] / 1e6,
            ],
            ax=ax[1],
            palette=custom_colors,
        )
        ax[1].set_title("Yearly OPEX", fontsize=24)
        ax[1].set(yticks=np.arange(0, 11, 1))

        for i in range(2):
            ax[i].set_ylabel("Million ($)")
            ax[i].tick_params(axis="x", rotation=45)

        return fig, ax
