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
color = ["#93003a", "#00429d", "#93c4d2", "#6ebf7c"]
matplotlib.rcParams.update({"legend.fontsize": 14, "legend.handlelength": 2})

from uam_system_model.FleetSizeOp import FleetSizeOptimizer


class FleetSizeOpSummary(FleetSizeOptimizer):
    def __init__(self, StarNetwork, policy):
        super().__init__(StarNetwork)

        self.specificc, self.specificu, self.specificn = self._parse_policy(policy)
        self.i = self.specificn["i"].unique()[1]
        self._compute_system_states()

    def get_summary_statistics(self):
        schedule = self.network.schedule.copy()

        fleet_size = (
            self.all_c.sum(axis=0)[0]
            + self.all_n.sum(axis=0)[0]
            + self.all_u.sum(axis=0)[0]
        )
        num_pads_at_hub = np.max(self.hub_c.sum(axis=0) + self.hub_n.sum(axis=0))
        num_pads_at_spoke = np.max(self.spoke_c.sum(axis=0) + self.spoke_n.sum(axis=0))
        total_number_of_flights = self.specificu["amount"].sum()

        i = self.specificn["i"].unique()[1]

        flight_demand = schedule[
            schedule["od"].isin(
                [
                    f"LAX_{self.network.vertiport_dict_inv[i]}",
                    f"{self.network.vertiport_dict_inv[i]}_LAX",
                ]
            )
        ].shape[0]
        repositioning_flights = total_number_of_flights - flight_demand

        energy_consumption = 0
        for idx, row in self.specificu.iterrows():
            t = int(row["t"])
            i = int(row["i"])
            j = int(row["j"])
            amount = int(row["amount"])
            energy_consumption += (
                amount * self.energy_consumption[t][i][j] * 0.025 * 160
            )  # Convert to kWh

        total_aircraft_miles = 0
        for idx, row in self.specificu.iterrows():
            i = int(row["i"])
            j = int(row["j"])
            amount = int(row["amount"])
            total_aircraft_miles += amount * self.network.flight_distance_matrix[i][j]

        revenue_aircraft_miles = (
            flight_demand * self.network.flight_distance_matrix[0][self.i]
        )

        summary = {
            "fleet_size": int(fleet_size),
            "total_pads": int(num_pads_at_hub + num_pads_at_spoke),
            "pads_at_hub": int(num_pads_at_hub),
            "pads_at_spoke": int(num_pads_at_spoke),
            "number_of_flights": int(total_number_of_flights),
            "number_of_repositioning_flights": int(repositioning_flights),
            "percentage_repositioning_flights": round(
                repositioning_flights / total_number_of_flights, 4
            ),
            "energy_consumption_kWh": round(energy_consumption, 2),
            "TAM": round(total_aircraft_miles, 2),
            "RAM": round(revenue_aircraft_miles, 2),
        }

        return summary

    def plot_aircraft_state(self):
        vertiport_name = " / " + self.network.vertiport_dict_inv[self.i]

        x0 = 0
        x1 = self.T
        fig, ax = plt.subplots(figsize=(12, 16), nrows=3, dpi=300)
        sns.lineplot(
            np.arange(x1),
            self.all_c.sum(axis=0)[x0:x1],
            label="Charging",
            ax=ax[0],
            color=color[0],
        )
        sns.lineplot(
            np.arange(x1),
            self.all_u.sum(axis=0)[x0:x1],
            label="In Flight",
            ax=ax[0],
            color=color[1],
        )
        sns.lineplot(
            np.arange(x1),
            self.all_n.sum(axis=0)[x0:x1],
            label="Idling",
            ax=ax[0],
            color=color[2],
        )
        sns.lineplot(
            np.arange(x1),
            (
                self.all_c.sum(axis=0)[x0:x1]
                + self.all_n.sum(axis=0)[x0:x1]
                + self.all_u.sum(axis=0)[x0:x1]
            ),
            color=color[3],
            label="All Aircraft",
            ax=ax[0],
        )

        sns.lineplot(
            np.arange(x1),
            self.hub_c.sum(axis=0)[x0:x1],
            label="Charging",
            ax=ax[1],
            color=color[0],
        )
        sns.lineplot(
            np.arange(x1),
            self.hub_u.sum(axis=0)[x0:x1],
            label="In Flight",
            ax=ax[1],
            color=color[1],
        )
        sns.lineplot(
            np.arange(x1),
            self.hub_n.sum(axis=0)[x0:x1],
            label="Idling",
            ax=ax[1],
            color=color[2],
        )
        sns.lineplot(
            np.arange(x1),
            (
                self.hub_c.sum(axis=0)[x0:x1]
                + self.hub_n.sum(axis=0)[x0:x1]
                + self.hub_u.sum(axis=0)[x0:x1]
            ),
            color=color[3],
            label="All Aircraft",
            ax=ax[1],
        )

        sns.lineplot(
            np.arange(x1),
            self.spoke_c.sum(axis=0)[x0:x1],
            label="Charging",
            ax=ax[2],
            color=color[0],
        )
        sns.lineplot(
            np.arange(x1),
            self.spoke_u.sum(axis=0)[x0:x1],
            label="In Flight",
            ax=ax[2],
            color=color[1],
        )
        sns.lineplot(
            np.arange(x1),
            self.spoke_n.sum(axis=0)[x0:x1],
            label="Idling",
            ax=ax[2],
            color=color[2],
        )
        sns.lineplot(
            np.arange(x1),
            (
                self.spoke_c.sum(axis=0)[x0:x1]
                + self.spoke_n.sum(axis=0)[x0:x1]
                + self.spoke_u.sum(axis=0)[x0:x1]
            ),
            color=color[3],
            label="All Aircraft",
            ax=ax[2],
        )

        ax[0].set(
            title="All Aircraft States",
            ylabel="Number of Aircrafts",
            xticks=np.concatenate([np.array([0, 1]), np.arange(24, 300, 12 * 2)]),
            xticklabels=[""] + [str(i) + ":00" for i in range(0, 25, 2)],
        )
        ax[1].set(
            title="Aircraft State at Hub",
            ylabel="Number of Aircrafts",
            xticks=np.concatenate([np.array([0, 1]), np.arange(24, 300, 12 * 2)]),
            xticklabels=[""] + [str(i) + ":00" for i in range(0, 25, 2)],
        )
        ax[2].set(
            title="Aircraft State at Spoke" + vertiport_name,
            ylabel="Number of Aircrafts",
            xticks=np.concatenate([np.array([0, 1]), np.arange(24, 300, 12 * 2)]),
            xticklabels=[""] + [str(i) + ":00" for i in range(0, 25, 2)],
        )
        for i in range(3):
            ax[i].set_xlabel("")
            ax[i].set_xlim([0, x1])
            ax[i].set_ylim([0, 10])
            ax[i].legend(loc="upper left", fontsize=14)
            ax[i].grid(True, linestyle="--", alpha=0.5)
            ax[i].xaxis.set_major_locator(MultipleLocator(12))
            ax[i].yaxis.set_major_locator(MultipleLocator(2))

        return fig, ax

    def _compute_system_states(self):
        self.all_c, self.all_u, self.all_n = self._compute_states(
            self.specificc, self.specificu, self.specificn
        )

        self.hub_c, self.hub_u, self.hub_n = self._compute_states(
            self.specificc[self.specificc["i"] == 0].reset_index(drop=True),
            self.specificu[self.specificu["i"] == 0].reset_index(drop=True),
            self.specificn[self.specificn["i"] == 0].reset_index(drop=True),
        )

        self.spoke_c, self.spoke_u, self.spoke_n = self._compute_states(
            self.specificc[self.specificc["i"] != 0].reset_index(drop=True),
            self.specificu[self.specificu["i"] != 0].reset_index(drop=True),
            self.specificn[self.specificn["i"] != 0].reset_index(drop=True),
        )

    def _parse_policy(self, policy):
        n = policy[policy["Variable"].str.contains("n")].reset_index(drop=True)
        u = policy[policy["Variable"].str.contains("u")].reset_index(drop=True)
        c = policy[policy["Variable"].str.contains("c")].reset_index(drop=True)

        n[["t", "i", "k"]] = (
            n["Variable"]
            .apply(lambda x: x.split("[")[1].replace("]", "").split(","))
            .tolist()
        )
        n["amount"] = n["Value"].astype(int)
        n = (
            n[["t", "i", "k", "amount"]]
            .astype(int)
            .sort_values(by=["t", "i", "k"])
            .reset_index(drop=True)
        )

        u[["t", "i", "j", "k"]] = (
            u["Variable"]
            .apply(lambda x: x.split("[")[1].replace("]", "").split(","))
            .tolist()
        )
        u["amount"] = u["Value"].astype(int)
        u = (
            u[["t", "i", "j", "k", "amount"]]
            .astype(int)
            .sort_values(by=["t", "i", "j", "k"])
            .reset_index(drop=True)
        )

        c[["t", "i", "x", "y"]] = (
            c["Variable"]
            .apply(lambda x: x.split("[")[1].replace("]", "").split(","))
            .tolist()
        )
        c["amount"] = c["Value"].astype(int)
        c = (
            c[["t", "i", "x", "y", "amount"]]
            .astype(int)
            .sort_values(by=["t", "i", "x", "y"])
            .reset_index(drop=True)
        )

        return c, u, n

    def _compute_states(self, specificc, specificu, specificn):
        end = int(1440 / self.time_step) + 1 + int(np.max(self.flight_time))
        all_c = np.zeros(shape=(1, end), dtype=int)
        for i in range(specificc.shape[0]):
            val = int(specificc["amount"][i])
            soc0 = int(specificc["x"][i])
            soc1 = int(specificc["y"][i])
            t = int(specificc["t"][i])
            time_charge = int(
                np.ceil(self.soc_transition_time[soc0:soc1].sum() / self.time_step)
            )
            occupied = np.zeros(shape=(val, end))
            for j in range(val):
                occupied[j][t : t + time_charge] = 1
            all_c = np.concatenate([all_c, occupied], axis=0)
        all_c = all_c[1:, :]

        all_u = np.zeros(shape=(1, end), dtype=int)
        for i in range(specificu.shape[0]):
            val = int(specificu["amount"][i])
            t = int(specificu["t"][i])
            origin = int(specificu["i"][i])
            dest = int(specificu["j"][i])
            flight = np.zeros(shape=(val, end))
            for j in range(val):
                flight[j][t : t + self.flight_time[t, origin, dest]] = 1
            all_u = np.concatenate([all_u, flight], axis=0)
        all_u = all_u[1:, :]

        all_n = np.zeros(shape=(1, end), dtype=int)
        for i in range(specificn.shape[0]):
            val = int(specificn["amount"][i])
            t = int(specificn["t"][i])
            idle = np.zeros(shape=(val, end))
            for j in range(val):
                idle[j][t] = 1
            all_n = np.concatenate([all_n, idle], axis=0)
        all_n = all_n[1:, :]

        return all_c, all_u, all_n
