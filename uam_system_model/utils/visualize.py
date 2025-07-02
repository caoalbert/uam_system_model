from matplotlib.ticker import MultipleLocator
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

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
    flight_count["origin"] = flight_count["od"].apply(lambda x: x.split("_")[0])
    flight_count["destination"] = flight_count["od"].apply(lambda x: x.split("_")[1])

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
    ax[0].set_title("APT-CBD")
    ax[1].set_title("CBD-APT")
    fig.text(
        0.08, 0.5, "Number of Flights", ha="center", va="center", rotation="vertical"
    )
    plt.legend(title="Vertiports", bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig, ax
