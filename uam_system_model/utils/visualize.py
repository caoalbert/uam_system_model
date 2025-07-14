from matplotlib.ticker import MultipleLocator
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns

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


def plot_hourly_flight_distribution(StarNetwork, ylim=(0, 25)):
    schedule = StarNetwork.schedule.copy()
    schedule["hour"] = schedule["schedule"] // 60
    schedule.loc[schedule["hour"] == 24.0, "hour"] = 0

    flight_count = schedule.groupby(["hour", "od"]).size().reset_index(name="count")
    pivot_table = flight_count.pivot_table(index='hour', columns='od', values='count', fill_value=0)
    flight_count = pivot_table.reset_index().melt(id_vars='hour', var_name='od', value_name='count')

    flight_count["origin"] = flight_count["od"].apply(lambda x: x.split("_")[0])
    flight_count["destination"] = flight_count["od"].apply(lambda x: x.split("_")[1])
    flight_count = flight_count.fillna(0)

    fig, ax = plt.subplots(figsize=(12, 4), ncols=2, dpi=200)
    for idx, i in enumerate(StarNetwork.vertiports[1:]):
        lax_cbd = flight_count[
            (flight_count["origin"] == "LAX") & (flight_count["destination"] == i)
        ]
        cbd_lax = flight_count[
            (flight_count["origin"] == i) & (flight_count["destination"] == "LAX")
        ]
        ax[0].plot(
            lax_cbd["hour"],
            lax_cbd["count"],
            color=color_palette[(idx) % 10],
            marker="o",
            label=StarNetwork.vertiports[idx + 1],
            linewidth=1,
        )
        ax[1].plot(
            cbd_lax["hour"],
            cbd_lax["count"],
            color=color_palette[(idx) % 10],
            marker="o",
            label=StarNetwork.vertiports[idx + 1],
            linewidth=1,
        )

    minorLocator = MultipleLocator(1)
    for i in range(2):
        ax[i].set(
            xlabel="",
            ylabel="",
            xticks=[0, 6, 12, 18, 24 - 1],
            xticklabels=["0:00", "6:00", "12:00", "18:00", "24:00"],
            ylim=ylim,
            xlim=(-0.2, 23.2),
        )
        ax[i].xaxis.set_minor_locator(minorLocator)
        ax[i].grid(True, alpha=0.25, linestyle="--", which="both")
    ax[0].set_title("LAX-Spokes")
    ax[1].set_title("Spokes-LAX")
    fig.text(
        0.08, 0.5, "Number of Flights", ha="center", va="center", rotation="vertical"
    )
    plt.legend(title="Vertiports", bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig, ax


def plot_parameters(StarNetwork):
    mask = np.triu(np.ones_like(StarNetwork.flight_distance_matrix, dtype=bool), k=1)

    fig = plt.figure(figsize=(26, 8), dpi=300)
    gs = gridspec.GridSpec(1, 6)  # 2 rows, 3 columns

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    sns.heatmap(
        StarNetwork.flight_distance_matrix,
        annot=True,
        cmap=cmap,
        ax=ax1,
        cbar_kws={"label": "Distance (Miles)"},
        mask=mask,
    )
    ax1.set(
        yticklabels=list(StarNetwork.vertiport_dict.keys()),
        xticklabels=list(StarNetwork.vertiport_dict.keys()),
        title="Distance",
    )
    ax1.invert_yaxis()
    plt.grid(False)

    sns.heatmap(
        StarNetwork.flight_time * 5,
        annot=True,
        cmap=cmap,
        ax=ax2,
        cbar_kws={"label": "Flight Time (min)"},
        mask=mask,
    )
    ax2.set(
        yticklabels=list(StarNetwork.vertiport_dict.keys()),
        xticklabels=list(StarNetwork.vertiport_dict.keys()),
        title="Flight Time",
    )
    ax2.invert_yaxis()
    plt.grid(False)

    sns.heatmap(
        StarNetwork.energy_consumption,
        annot=True,
        cmap=cmap,
        ax=ax3,
        cbar_kws={"label": "Energy Consumption (% SoC)"},
        mask=mask,
    )
    ax3.set(
        yticklabels=list(StarNetwork.vertiport_dict.keys()),
        xticklabels=list(StarNetwork.vertiport_dict.keys()),
        title="Energy Consumption",
    )
    ax3.invert_yaxis()

    for ax in [ax1, ax2, ax3]:
        ax.title.set_fontsize(30)
    plt.tight_layout()

    return fig, (ax1, ax2, ax3)

def plot_travel_time(travel_time_avg, 
                     vertiports, 
                     ylim=(0, 100),
                     ylabel="TNC Trip Time (min)"):

    fig, ax = plt.subplots(figsize=(12, 4), ncols=2, dpi=200)
    for i in range(1,9):
        sns.lineplot(
            x=np.arange(24),
            y=travel_time_avg[0, i, :],
            label=vertiports[i],
            ax=ax[0],
            color=color_palette[i-1],
            linewidth=1,
            marker='o',
            legend=False,
        )
        sns.lineplot
        sns.lineplot(
            x=np.arange(24),
            y=travel_time_avg[i, 0, :],
            label=vertiports[i],
            ax=ax[1],
            color=color_palette[i-1],
            linewidth=1,
            marker='o'
        )

    minorLocator = MultipleLocator(1)
    for i in range(2):
        ax[i].set(
            xlabel="",
            ylabel="",
            xticks=[0, 6, 12, 18, 24 - 1],
            xticklabels=["0:00", "6:00", "12:00", "18:00", "24:00"],
            xlim=(-0.2, 23.2),
            ylim= ylim,
        )
        ax[i].xaxis.set_minor_locator(minorLocator)
        ax[i].grid(True, alpha=0.25, linestyle="--", which="both")
    ax[0].set_title("LAX-Spokes")
    ax[1].set_title("Spokes-LAX")
    fig.text(
        0.07, 0.5, ylabel, ha="center", va="center", rotation="vertical"
    )
    plt.legend(title="Vertiports", bbox_to_anchor=(1.05, 1), loc="upper left");

    return fig, ax