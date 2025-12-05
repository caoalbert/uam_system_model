import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ScheduleGeneratorBJ import ScheduleGenerator
from .utils.schedule_utils import generate_uam_schedule
from .utils.visualize import plot_travel_time

DATA_PATH_1 = os.path.join(
    os.path.dirname(__file__), "data", "PEK-PKX-Schedules20251023.xlsx"
)


class StarNetwork:
    """
    北京专用：双 Hub（PEK / PKX）+ 多个 spoke 的星型网络。

    约定：
        - vertiport_names 的顺序与各类矩阵的行列顺序一致
        - 其中包含 "PEK" 与 "PKX" 两个 hub，其余视为 spoke
    """

    def __init__(
        self,
        vertiport_names: list,
        flight_distance_matrix: np.ndarray,
        flight_time_matrix: np.ndarray,
        energy_consumption_matrix: np.ndarray,
    ):
        # 基础网络信息
        self.vertiports = list(vertiport_names)
        self.vertiport_dict = {name: idx for idx, name in enumerate(self.vertiports)}
        self.vertiport_dict_inv = {
            idx: name for name, idx in self.vertiport_dict.items()
        }

        self.flight_distance_matrix = flight_distance_matrix
        self.flight_time = flight_time_matrix
        self.energy_consumption = energy_consumption_matrix

        # 读 Excel 的调度生成器
        self.sched = ScheduleGenerator(DATA_PATH_1)

        # hub / spoke 信息
        self.hub_set = {"PEK", "PKX"}
        self.spoke_indices = [
            idx
            for idx, name in self.vertiport_dict_inv.items()
            if name not in self.hub_set
        ]

        # 运行结果容器
        self.schedule: pd.DataFrame | None = None
        self.pax_arrival_times: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    #  需求加载与调度生成
    # ------------------------------------------------------------------
    def load_demand(
        self,
        month: int,
        day: int,
        hub_names,
        vertiport_pmf: np.ndarray,
        directional_demand: int,
        auto_regressive_alpha: float = 0.0,
        max_waiting_time: int = 5,
        occupancy: int = 4,
        fare: float = 3.0,
        seed: int = 0,
    ):
        """
        对于 hub_names 中的每一个 hub（例如 ["PEK", "PKX"]）：
            1. 从 Excel 中抽取该 hub 的到离港时刻；
            2. 构造“以当前 hub 为索引 0”的局部网络视角；
            3. 调用 schedule_utils.generate_uam_schedule 生成单 hub 调度；
            4. 将多个 hub 的结果拼接起来。
        """
        self.month = month
        self.day = day

        np.random.seed(seed)

        hub_names = list(hub_names)

        all_schedule = []
        all_pax = []

        n_vertiports = len(self.vertiports)

        for hub_name in hub_names:
            if hub_name not in self.vertiport_dict:
                raise ValueError(f"未知 hub: {hub_name}")

            # 1) 从 Excel 获取该 hub 的到离港时刻（带 capacity）
            arr_df, dep_df, yearly_capacity = self.sched.get_one_day_for_hub(
                month=month, day=day, hub_name=hub_name
            )

            # 2) 构造“局部视角”：当前 hub 放在 index 0
            hub_global_idx = self.vertiport_dict[hub_name]
            perm = [hub_global_idx] + [
                i for i in range(n_vertiports) if i != hub_global_idx
            ]

            # 局部的 index → 名字 映射
            vertiport_dict_inv_local = {
                i: self.vertiport_dict_inv[perm[i]] for i in range(n_vertiports)
            }

            # 局部距离矩阵
            flight_distance_matrix_local = self.flight_distance_matrix[
                np.ix_(perm, perm)
            ]

            # 局部 pmf：对 perm 重新排序后再归一化
            vertiport_pmf_local_raw = vertiport_pmf[perm].astype(float)
            total_prob = vertiport_pmf_local_raw.sum()
            if total_prob <= 0:
                raise ValueError("vertiport_pmf 的总和必须大于 0")
            vertiport_pmf_local = vertiport_pmf_local_raw / total_prob

            # 3) 调用单 hub 的黑盒调度函数（不改 util）
            schedule_h, pax_h = generate_uam_schedule(
                hub_flight_arr=arr_df,
                hub_flight_dep=dep_df,
                total_hub_seats=yearly_capacity,
                auto_regressive_alpha=auto_regressive_alpha,
                directional_demand=directional_demand,
                vertiport_pmf=vertiport_pmf_local,
                occupancy=occupancy,
                max_waiting_time=max_waiting_time,
                vertiport_dict_inv=vertiport_dict_inv_local,
                hub_name=hub_name,
                flight_distance_matrix=flight_distance_matrix_local,
                fare=fare,
            )

            all_schedule.append(schedule_h)
            all_pax.append(pax_h)

        schedule = pd.concat(all_schedule, ignore_index=True)
        schedule["revenue"] = (
            schedule["revenue"]
            / schedule["num_pax"]
            / 3
            * 1.6
            * 2.62
            * schedule["num_pax"]
            / 7.07
        )

        self.schedule = schedule
        self.pax_arrival_times = pd.concat(all_pax, ignore_index=True)

        return self.schedule, self.pax_arrival_times

    def plot_flight(self, ylim=(0, 25)):

        # 假定：前两个是 hub，后面都是 spoke
        hubs = self.vertiports[:2]
        spokes = self.vertiports[2:]

        schedule = self.schedule.copy()
        schedule["origin"] = schedule["od"].str.split("_").str[0]
        schedule["destination"] = schedule["od"].str.split("_").str[1]
        schedule["hour"] = (schedule["schedule"] // 60).astype(int)
        schedule.loc[schedule["hour"] == 24, "hour"] = 0

        figs = []
        axes_list = []

        for hub_name in hubs:
            # 只保留与当前 hub 有关的航班（hub <-> 任意点）
            sub = schedule[
                (schedule["origin"] == hub_name)
                | (schedule["destination"] == hub_name)
            ].copy()

            # 构造“局部视角”：
            #   [hub] + [7 个 spokes] + [1 个 dummy 占位]
            local_vertiports = [hub_name] + spokes + ["_nolegend_"]
            n_local = len(local_vertiports)  # 应该是 9

            # 名称 -> 局部索引
            local_idx = {name: idx for idx, name in enumerate(local_vertiports)}

            # 1. 先全部置为 0（没航班=0，线是连续的）
            input_to_viz = np.zeros((n_local, n_local, 24), dtype=float)

            for _, row in sub.iterrows():
                o = row["origin"]
                d = row["destination"]
                h = int(row["hour"])

                if (o == hub_name and d in spokes) or (d == hub_name and o in spokes):
                    i = local_idx[o]
                    j = local_idx[d]
                else:
                    continue

                input_to_viz[i, j, h] += 1.0

            # 2. 把 dummy 端点那一行/一列设成 nan，这样不会画出这条全是0的线
            dummy_idx = local_idx["_nolegend_"]
            input_to_viz[dummy_idx, :, :] = np.nan
            input_to_viz[:, dummy_idx, :] = np.nan


            # 调用底层原始的 plot_travel_time（不改它）
            fig, ax = plot_travel_time(
                input_to_viz,
                local_vertiports,
                ylim=ylim,
                ylabel="Number of Flights",
            )

            # --- 覆盖 plot_travel_time 的标题 ---
            hub_short = hub_name   # 例如 "PEK", "PKX"

            ax[0].set_title(f"{hub_short}-Spokes")    # 左图：hub → spokes
            ax[1].set_title(f"Spokes-{hub_short}")    # 右图：spokes → hub

            figs.append(fig)
            axes_list.append(ax)

        return figs, axes_list

    def plot_pax(self, ylim=(0, 25)):
        """
        使用已有的 plot_travel_time，将日均每小时 OD 乘客数量可视化。
        """
        if self.pax_arrival_times is None:
            raise RuntimeError("请先调用 load_demand 生成 pax 到达过程")

        pax_df = self.pax_arrival_times.copy()
        pax_df["hour"] = (pax_df["passenger_arrival_time_s"] // 3600).astype(int)

        pax_count = (
            pax_df.groupby(["origin_vertiport_id", "destination_vertiport_id", "hour"])
            .size()
            .reset_index(name="counts")
        )

        input_to_viz = np.zeros((len(self.vertiports), len(self.vertiports), 24))
        for _, row in pax_count.iterrows():
            o = int(row["origin_vertiport_id"])
            d = int(row["destination_vertiport_id"])
            h = int(row["hour"])
            input_to_viz[o, d, h] = row["counts"]

        fig, ax = plot_travel_time(
            input_to_viz, self.vertiports, ylim=ylim, ylabel="Number of Passengers"
        )

        return fig, ax
