import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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

# import pandas as pd


class PricingOpSummary(PricingOptimizer):
    def __init__(self, StarNetwork, policy):
        super().__init__(StarNetwork)
        self.policy = policy

    #     all_routing = pd.DataFrame({'passenger_arrival_time_slot': np.repeat(np.arange(49), 16),
    #                                 'origin_vertiport_id': np.tile(np.concatenate((np.repeat(0, 8), np.arange(1, 9))), 49),
    #                                 'destination_vertiport_id': np.tile(np.concatenate((np.arange(1, 9),np.repeat(0, 8))), 49),
    #                             })
    #     all_routing = pd.merge(policy, all_routing, on=['passenger_arrival_time_slot', 'origin_vertiport_id', 'destination_vertiport_id'], how='outer')

    #     all_routing['markets'] = all_routing.apply(self.identify_market, axis=1)

    #     self.policy = all_routing

    # def identify_market(self, row):
    #     if row['origin_vertiport_id'] == 0:
    #         return self.network.vertiport_dict_inv[row['destination_vertiport_id']]
    #     else:
    #         return self.network.vertiport_dict_inv[row['origin_vertiport_id']]

    def get_summary_statistics(self):
        summary_stats = {}

        summary_stats["total_revenue"] = self.policy["total_revenue"].sum()
        summary_stats["total_num_flights"] = self.policy["num_flights"].sum()

        return summary_stats

    def plot_average_rasm(self, dpi=300):
        df_grouped = self.policy.groupby(["passenger_arrival_time_slot"])[
            "rev_per_mile"
        ].mean()
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
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
            title="Average RASM over ODs",
        )
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(True, linestyle="--", alpha=0.5)

        return fig, ax

    def plot_rasm_by_od(self, dpi=300):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        sns.lineplot(
            data=self.policy,
            x="passenger_arrival_time_slot",
            hue="markets",
            hue_order=self.network.vertiports[1:],
            y="rev_per_mile",
            marker="o",
            err_style=None,
            palette=custom_colors,
            ax=ax,
        )
        ax.set(
            xticks=np.arange(0, 49, 12),
            xticklabels=[str(i) + ":00" for i in range(0, 26, 6)],
            xlabel="",
            ylabel="RASM ($)",
            xlim=(0, 48),
            title="Average RASM by ODs",
        )
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 0.85), loc="upper left", borderaxespad=0.0)

        return fig, ax

    def plot_fare_by_od(self, dpi=300):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        sns.lineplot(
            data=self.policy,
            x="passenger_arrival_time_slot",
            y="fare",
            hue="markets",
            hue_order=self.network.vertiports[1:],
            err_style=None,
            legend=True,
            marker="o",
            palette=custom_colors,
            ax=ax,
        )
        ax.set(
            xticks=np.arange(0, 49, 12),
            xticklabels=[str(i) + ":00" for i in range(0, 26, 6)],
            xlabel="",
            ylabel="Fare ($)",
            xlim=(0, 48),
            title="Average Fare by ODs",
        )
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 0.85), loc="upper left", borderaxespad=0.0)

        return fig, ax

    def plot_revenue_by_od(self, dpi=300):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        sns.lineplot(
            data=self.policy,
            x="passenger_arrival_time_slot",
            y="total_revenue",
            hue="markets",
            hue_order=self.network.vertiports[1:],
            err_style=None,
            legend=True,
            marker="o",
            palette=custom_colors,
            ax=ax,
        )
        ax.set(
            xticks=np.arange(0, 49, 12),
            xticklabels=[str(i) + ":00" for i in range(0, 26, 6)],
            xlabel="",
            ylabel="Revenue ($)",
            xlim=(0, 48),
            title="Total Revenue by ODs",
        )
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 0.85), loc="upper left", borderaxespad=0.0)

        return fig, ax

    def plot_uam_share_by_od(self, dpi=300):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        sns.lineplot(
            data=self.policy,
            x="passenger_arrival_time_slot",
            y="percentage_uam",
            hue="markets",
            hue_order=self.network.vertiports[1:],
            err_style=None,
            legend=True,
            marker="o",
            palette=custom_colors,
            ax=ax,
        )
        ax.set(
            xticks=np.arange(0, 49, 12),
            xticklabels=[str(i) + ":00" for i in range(0, 26, 6)],
            xlabel="",
            ylabel=r"$P(U^{UAM} > U^{TNC})$",
            xlim=(0, 48),
            title="UAM Market Share by ODs",
        )
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 0.85), loc="upper left", borderaxespad=0.0)

        return fig, ax
