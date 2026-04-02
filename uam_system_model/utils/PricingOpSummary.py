import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

sns.set(style="whitegrid", font_scale=1)

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
color = ["#93003a", "#00429d", "#93c4d2", "#6ebf7c"]
matplotlib.rcParams.update({"legend.fontsize": 14, "legend.handlelength": 2})

from uam_system_model.Pricing import PricingOptimizer


class PricingOpSummary(PricingOptimizer):
    def __init__(self, StarNetwork, policy):
        super().__init__(StarNetwork)
        slots = pd.DataFrame(
            {
                "passenger_arrival_time_slot": policy[
                    "passenger_arrival_time_slot"
                ].unique()
            }
        )
        origins = pd.DataFrame({"origin": ["apt_dt", "dt_apt"]})
        destinations = pd.DataFrame(
            {"destination": list(self.network.vertiport_dict.values())[1:]}
        )

        all_combos = slots.merge(origins, how="cross").merge(destinations, how="cross")
        all_combos["origin_vertiport_id"] = all_combos.apply(
            lambda row: 0 if row["origin"] == "apt_dt" else row["destination"], axis=1
        )
        all_combos["destination_vertiport_id"] = all_combos.apply(
            lambda row: row["destination"] if row["origin"] == "apt_dt" else 0, axis=1
        )
        all_combos = all_combos[
            [
                "passenger_arrival_time_slot",
                "origin_vertiport_id",
                "destination_vertiport_id",
            ]
        ]

        policy = pd.merge(
            all_combos,
            policy,
            on=[
                "passenger_arrival_time_slot",
                "origin_vertiport_id",
                "destination_vertiport_id",
            ],
            how="outer",
        )
        policy["markets"] = policy.apply(
            lambda row: self.network.vertiport_dict_inv[row["destination_vertiport_id"]]
            if row["origin_vertiport_id"] == 0
            else self.network.vertiport_dict_inv[row["origin_vertiport_id"]],
            axis=1,
        )
        self.policy = policy

    def get_summary_statistics(self):
        summary_stats = {}

        summary_stats["total_revenue"] = self.policy["total_revenue"].sum()
        summary_stats["total_num_flights"] = self.policy["num_flights"].sum()

        return summary_stats

    def plot_average_rasm(self, ax=None, ylim=(0, 12), dpi=300):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2.5), dpi=dpi)
        else:
            fig = ax.figure

        df_grouped = self.policy.groupby(["passenger_arrival_time_slot"])[
            "rev_per_mile"
        ].mean()

        sns.lineplot(
            x=np.arange(len(df_grouped)),
            y=df_grouped.values,
            marker="o",
            color=color[1],
            ax=ax,
        )
        ax.set(
            xticks=np.arange(0, 49, 12),
            xticklabels=[str(i) + ":00" for i in range(0, 26, 6)],
            xlabel="",
            ylabel="RASM ($)",
            xlim=(0, 48),
            ylim=ylim,
            title="Average RASM over Passengers",
        )
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.grid(True, which="major", linestyle="--", alpha=0.6, linewidth=1)
        ax.grid(True, which="minor", linestyle="--", alpha=0.2, linewidth=1)

        return fig, ax

    def plot_rasm_by_od(self, ax=None, dpi=300, ylim=(0, 12)):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        else:
            fig = ax.figure

        markets_order = self.network.vertiports[1:]

        for i, market in enumerate(markets_order):
            market_data = self.policy[self.policy["markets"] == market]
            market_data_agg = (
                market_data.groupby("passenger_arrival_time_slot")["rev_per_mile"]
                .mean()
                .reset_index()
                .sort_values("passenger_arrival_time_slot")
            )
            color = (
                custom_colors[i]
                if isinstance(custom_colors, list)
                else custom_colors.get(market)
            )
            ax.plot(
                market_data_agg["passenger_arrival_time_slot"],
                market_data_agg["rev_per_mile"],
                label=market,
                marker="o",
                markeredgecolor="white",
                color=color,
            )

        ax.set(
            xticks=np.arange(0, 49, 12),
            xticklabels=[str(i) + ":00" for i in range(0, 26, 6)],
            xlabel="",
            ylabel="Fare per mile ($/mile)",
            xlim=(0, 48),
            title="Average Fare per Mile by Market",
            ylim=ylim,
        )
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.grid(True, which="major", linestyle="--", alpha=0.6, linewidth=1)
        ax.grid(True, which="minor", linestyle="--", alpha=0.2, linewidth=1)
        ax.legend(bbox_to_anchor=(1.02, 0.85), loc="upper left", borderaxespad=0.0)

        return fig, ax

    def plot_fare_by_od(self, ax=None, dpi=300):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        else:
            fig = ax.figure

        markets_order = self.network.vertiports[1:]

        for i, market in enumerate(markets_order):
            market_data = self.policy[self.policy["markets"] == market]
            market_data_agg = (
                market_data.groupby("passenger_arrival_time_slot")["fare"]
                .mean()
                .reset_index()
                .sort_values("passenger_arrival_time_slot")
            )

            color = (
                custom_colors[i]
                if isinstance(custom_colors, list)
                else custom_colors.get(market)
            )
            ax.plot(
                market_data_agg["passenger_arrival_time_slot"],
                market_data_agg["fare"],
                label=market,
                marker="o",
                markeredgecolor="white",
                color=color,
            )

        ax.set(
            xticks=np.arange(0, 49, 12),
            xticklabels=[str(i) + ":00" for i in range(0, 26, 6)],
            xlabel="",
            ylabel="Fare ($)",
            xlim=(0, 48),
            title="Average Fare by Market",
        )
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.grid(True, which="major", linestyle="--", alpha=0.6, linewidth=1)
        ax.grid(True, which="minor", linestyle="--", alpha=0.2, linewidth=1)
        ax.legend(bbox_to_anchor=(1.02, 0.85), loc="upper left", borderaxespad=0.0)

        return fig, ax

    def plot_revenue_by_od(self, ax=None, dpi=300, ylim=(0, 2000)):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        else:
            fig = ax.figure

        markets = self.network.vertiports[1:]
        for i, market in enumerate(markets):
            subset = self.policy[self.policy["markets"] == market]
            if isinstance(custom_colors, dict):
                color = custom_colors.get(market)
            else:
                color = custom_colors[i % len(custom_colors)]

            ax.plot(
                subset["passenger_arrival_time_slot"],
                subset["total_revenue"],
                marker="o",
                markeredgecolor="white",
                linestyle="-",  # Explicitly add the line (seaborn does this by default)
                color=color,
                label=str(market),  # Assign the label so ax.legend() picks it up
            )
        ax.set(
            xticks=np.arange(0, 49, 12),
            xticklabels=[str(i) + ":00" for i in range(0, 26, 6)],
            xlabel="",
            ylabel="Revenue ($)",
            xlim=(0, 48),
            ylim=ylim,
            title="Total Revenue by ODs",
        )
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.grid(True, which="major", linestyle="--", alpha=0.6, linewidth=1)
        ax.grid(True, which="minor", linestyle="--", alpha=0.2, linewidth=1)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

        return fig, ax

    def plot_uam_share_by_od(self, ax=None, dpi=300):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        else:
            fig = ax.figure

        markets_order = self.network.vertiports[1:]

        for i, market in enumerate(markets_order):
            market_data = self.policy[self.policy["markets"] == market]
            market_data_agg = (
                market_data.groupby("passenger_arrival_time_slot")["percentage_uam"]
                .mean()
                .reset_index()
            )

            color = (
                custom_colors[i]
                if isinstance(custom_colors, list)
                else custom_colors.get(market)
            )
            ax.plot(
                market_data_agg["passenger_arrival_time_slot"],
                market_data_agg["percentage_uam"],
                label=market,
                marker="o",
                color=color,
            )

        ax.set(
            xticks=np.arange(0, 49, 12),
            xticklabels=[str(i) + ":00" for i in range(0, 26, 6)],
            xlabel="",
            ylabel=r"$P(U^{UAM} > U^{TNC})$",
            xlim=(0, 48),
            title="UAM Market Share by ODs",
        )
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))

        ax.grid(
            True, which="major", color="gray", linestyle="--", alpha=0.6, linewidth=0.5
        )
        ax.grid(
            True, which="minor", color="gray", linestyle="--", alpha=0.2, linewidth=0.5
        )
        ax.legend(bbox_to_anchor=(1.02, 0.95), loc="upper left", borderaxespad=0.0)

        return fig, ax
