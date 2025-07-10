from gurobipy import *
import pandas as pd
import numpy as np
import ast
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

font = {"size": 24}
matplotlib.rc("font", **font)
import warnings

warnings.filterwarnings("ignore")


class FleetSizeOptimizer:
    def __init__(
        self,
        StarNetwork,
        fixed_cost=1,
        variable_cost=0.00001,
        time_step=5,
    ):
        self.network = StarNetwork

        # Load flight time
        self.flight_time = self.network.flight_time

        # Prepare demand
        schedule = StarNetwork.schedule.copy()
        schedule["schedule"] = schedule["schedule"] // 5
        schedule["origin"] = schedule["od"].apply(
            lambda x: self.network.vertiport_dict[x.split("_")[0]]
        )
        schedule["destination"] = schedule["od"].apply(
            lambda x: self.network.vertiport_dict[x.split("_")[1]]
        )
        schedule = (
            schedule.groupby(["schedule", "origin", "destination"])
            .agg(
                count=("origin", "count"),
                revenue=("revenue", "sum"),
            )
            .reset_index()
            .rename(columns={"schedule": "time"})
        )
        schedule["per_flight_revenue"] = schedule["revenue"] / schedule["count"]
        self.flight_demand = schedule


        # Set up time-related parameters
        self.time_step = time_step
        self.schedule_time_step = int(1440 / time_step)
        self.soc_transition_time = (
            np.array(
                [
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
        self.energy_consumption = np.ceil(
            self.network.energy_consumption / (80 / len(self.soc_transition_time))
        ).astype(int)

        # Check the dimension of flight time and energy consumption. Make sure they are time-varying.
        if len(self.flight_time.shape) != len(self.energy_consumption.shape):
            raise DimensionError(
                "The dimension of the flight time array must match the dimension of the energy consumption array"
            )
        if len(self.flight_time.shape) == 2:
            self.flight_time = np.repeat(
                self.flight_time[np.newaxis, :, :], 
                self.schedule_time_step + 1, 
                axis=0
            )
        if len(self.energy_consumption.shape) == 2:
            self.energy_consumption = np.repeat(
                self.energy_consumption[np.newaxis, :, :],
                self.schedule_time_step + 1,
                axis=0,
            )

        # Set up fixed and variable cost
        self.fixed_cost = fixed_cost
        self.variable_cost = variable_cost

        # Get constants
        self.K = len(self.soc_transition_time)
        self.T = int(np.max(self.flight_time)) + self.schedule_time_step + 1

    @staticmethod
    def get_2_vertiport_demand(schedule, vertiport_index):
        APT_CBD = np.zeros(shape=(288,), dtype=int)
        CBD_APT = np.zeros(shape=(288,), dtype=int)
        for time in range(1, 289):
            schedule_at_time = schedule[schedule["time"] == time]
            for idx, row in schedule_at_time.iterrows():
                origin = row["origin"]
                destination = row["destination"]
                count = row["count"]
                if (origin == 0) & (destination == vertiport_index):
                    APT_CBD[time-1] += count
                elif (destination == 0) & (origin == vertiport_index):
                    CBD_APT[time-1] += count

        return (APT_CBD, CBD_APT)
    

    def optimize(
        self,
        vertiport_index,
        charging_station=[True, True],
        number_of_pads=None,
        verbose=True,
        optimality_gap=0.05,
        n_threads=24,
    ):
        """
        param vertiport_index: The index of the secondary vertiport. The primary vertiport is always indexed as 0.
        param charging_station: A binary list indicating whether or not charging infrastructure exists at each vertiport.
        param number_of_pads: A list indicating the number of parking pads at each vertiport. If None, then no parking constraint is applied.
        param verbose: Whether or not to print out the optimization log.
        param optimality_gap: The optimality gap for the optimization solver.
        param n_threads: Number of threads to use for optimization.
        """

        # Load the parameters
        T = self.T
        K = self.K
        SELECT_VERTIPORT_INDEX = [0, vertiport_index]
        gamma = self.soc_transition_time

        f_values = np.zeros((self.T, len(self.network.vertiports), len(self.network.vertiports)))
        LAX_DTLA, DTLA_LAX = self.get_2_vertiport_demand(self.flight_demand, vertiport_index)
        for t in range(self.T - self.flight_time.max() - 1):
            f_values[t + 1][0][vertiport_index] = LAX_DTLA[t]
            f_values[t + 1][vertiport_index][0] = DTLA_LAX[t]
            

        tau = []
        kappa = []
        for _ in range(T):
            tau_second_layer = []
            kappa_second_layer = []
            for _ in range(len(self.network.vertiports)):
                tau_third_layer = []
                kappa_third_layer = []
                for _ in range(len(self.network.vertiports)):
                    tau_innermost_layer = [0]
                    kappa_innermost_layer = [0]
                    tau_third_layer.append(tau_innermost_layer)
                    kappa_third_layer.append(kappa_innermost_layer)

                tau_second_layer.append(tau_third_layer)
                kappa_second_layer.append(kappa_third_layer)
            tau.append(tau_second_layer)
            kappa.append(kappa_second_layer)

        # tau here is a mapping from t to the time an aircraft leaves from the origin
        for t in range(1, self.schedule_time_step + 1):
            for i in SELECT_VERTIPORT_INDEX:
                for j in SELECT_VERTIPORT_INDEX:
                    if i != j:
                        tau[t + self.flight_time[t, i, j]][i][j].append(t)
                        kappa[t + self.flight_time[t, i, j]][i][j].append(
                            self.energy_consumption[t, i, j]
                        )

        ########################################Optimization Model##############################################
        m = Model("FleetSizeOptimizer")
        if verbose == False:
            m.setParam("OutputFlag", 0)
        m.setParam("threads", n_threads)

        # Create variables
        ni = m.addVars(
            ((t, i, k) for t in range(T) for i in SELECT_VERTIPORT_INDEX for k in range(K + 1)),
            vtype=GRB.INTEGER,
            name="n",
        )
        uij = m.addVars(
            (
                (t, i, j, k)
                for t in range(T)
                for i in SELECT_VERTIPORT_INDEX
                for j in SELECT_VERTIPORT_INDEX
                for k in range(K + 1)
                if i != j
            ),
            vtype=GRB.INTEGER,
            name="u",
        )
        cijk = m.addVars(
            (
                (t, i, x, y)
                for t in range(T)
                for i in SELECT_VERTIPORT_INDEX
                for x in range(K + 1)
                for y in range(K + 1)
                if x < y
            ),
            vtype=GRB.INTEGER,
            name="c",
        )

        m.setObjective(
            self.fixed_cost
            * (
                ni.sum(0, "*", "*")
                + uij.sum(0, "*", "*", "*")
                + cijk.sum(0, "*", "*", "*")
            )
            + self.variable_cost * (uij.sum("*", "*", "*", "*")),
            GRB.MINIMIZE,
        )

        # Constraint 1: Dynamic equation
        for i in SELECT_VERTIPORT_INDEX:
            for k in range(K + 1):
                for t in range(1, T):
                    m.addConstr(
                        ni[t, i, k]
                        == ni[t - 1, i, k]
                        + quicksum(
                            uij[tau[t][j][i][index], j, i, k + kappa[t][j][i][index]]
                            for j in SELECT_VERTIPORT_INDEX
                            for index in range(len(tau[t][j][i]))
                            if j != i
                            and tau[t][j][i][index] != 0
                            and k + kappa[t][j][i][index] <= K
                        )
                        - quicksum(uij[t, i, j, k] for j in SELECT_VERTIPORT_INDEX if j != i)
                        + quicksum(
                            cijk[t - np.ceil(sum(gamma[x:k]) / self.time_step), i, x, k]
                            for x in range(k)
                            if t - np.ceil(sum(gamma[x:k]) / 1) >= 0
                        )
                        - quicksum(cijk[t, i, k, y] for y in range(k + 1, K + 1))
                    )

        # Constraint 2: Cyclo-stationarity Constraint
        for k in range(K + 1):
            for i in SELECT_VERTIPORT_INDEX:
                for j in SELECT_VERTIPORT_INDEX:
                    if i != j:
                        m.addConstr(uij[0, i, j, k] == uij[T - 1, i, j, k])
                m.addConstr(ni[0, i, k] == ni[T - 1, i, k])

        for x in range(K + 1):
            for y in range(K + 1):
                for i in SELECT_VERTIPORT_INDEX:
                    if x < y:
                        m.addConstr(cijk[0, i, x, y] == cijk[T - 1, i, x, y])

        # Constraint 3: Demand Constraint
        for j in SELECT_VERTIPORT_INDEX:
            for i in SELECT_VERTIPORT_INDEX:
                if i != j:
                    for t in range(T - 1):
                        m.addConstr(uij.sum(t, i, j, "*") >= f_values[t][i][j])

        # Constraint 4: Energy Constraint
        for i in SELECT_VERTIPORT_INDEX:
            for j in SELECT_VERTIPORT_INDEX:
                for t in range(T):
                    if i != j:
                        m.addConstr(uij[t, i, j, 0] == 0)

        # Constraint 5: Charging Station Constraint
        for idx, i in enumerate(charging_station):
            if i == False:
                m.addConstr(cijk.sum("*", SELECT_VERTIPORT_INDEX[idx], "*", "*") == 0)

        row, col, TPRIME = self.__getVars__()

        # Constraint 6: Parking Facility Constraint
        if number_of_pads is not None:
            for i in SELECT_VERTIPORT_INDEX:
                for t in range(T):
                    m.addConstr(
                        (
                            ni.sum(t, i, "*")
                            + quicksum(
                                cijk[time_adjusted, i, row[index], col[index]]
                                for index in range(len(col))
                                for time_adjusted in range(
                                    t - int(TPRIME[row[index], col[index]]), t
                                )
                                if t >= TPRIME[row[index], col[index]]
                            )
                            + cijk.sum(t, i, "*", "*")
                        )
                        <= number_of_pads[i]
                    )

        m.update()
        m.Params.MIPGap = optimality_gap
        m.Params.FeasibilityTol = 1e-5
        m.optimize()

        total_fleet_size = int(
            ni.sum(0, "*", "*").getValue()
            + uij.sum(0, "*", "*", "*").getValue()
            + cijk.sum(0, "*", "*", "*").getValue()
        )

        results = []
        for v in m.getVars():
            if v.x > 1e-6: 
                results.append({
                    'Variable': v.varName,
                    'Value': v.x
                })
        results_df = pd.DataFrame(results)

        return total_fleet_size, results_df

    def __getVars__(self):
        w = np.zeros(shape=(self.K + 1, self.K + 1))
        for i in range(self.K + 1):
            for j in range(self.K + 1):
                if j > i:
                    w[i, j] = self.soc_transition_time[i:j].sum()
        w = w // 5 + 1
        row, col = np.where(w > 1)

        return row, col, w - 1