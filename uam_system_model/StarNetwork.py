import numpy as np
from .ScheduleGenerator import ScheduleGenerator
from .utils.visualize import plot_travel_time

class StarNetwork:
    """
    北京专用：双 Hub（索引 0=PEK, 1=PKX），其余 7 个为 spoke。
    """

    def __init__(
        self,
        vertiport_names: list,                 # ["PEK","PKX","CBD","ZGC","WJ","TON","LZC","SLT","HDX"]
        flight_distance_matrix: np.ndarray,    
        flight_time_matrix: np.ndarray,        
        energy_consumption_matrix: np.ndarray, 
        path_schedule: str,                    # 北京 Excel 文件路径（放在原 LAX_ind.csv 的同位置）
    ):
        assert len(vertiport_names) == 9, "北京版本期望 9 个节点（2 hub + 7 spoke）"
        n = len(vertiport_names)

        self.vertiports = vertiport_names
        self.vertiport_dict = {name: idx for idx, name in enumerate(vertiport_names)}
        self.vertiport_dict_inv = {idx: name for idx, name in enumerate(vertiport_names)}
        self.hubs = [0, 1]  # 写死为北京双 hub：0=PEK, 1=PKX

        assert flight_distance_matrix.shape == (n, n)
        assert flight_time_matrix.shape == (n, n)
        assert energy_consumption_matrix.shape == (n, n)

        self.flight_distance_matrix = flight_distance_matrix
        self.flight_time = flight_time_matrix
        self.energy_consumption = energy_consumption_matrix

        # 北京专用的时刻解析器（Excel 的四个 sheet）
        self.sched = ScheduleGenerator(path_schedule)

        # 输出占位
        self.schedule = None
        self.pax_arrival_times = None

    def load_demand(
        self,
        month: int,                 
        day: int,                   
        vertiport_pmf: np.ndarray,  
        directional_demand: int,    # 仅当 pmf 为 1D 时需要
        auto_regressive_alpha: float = 0.0,
        max_waiting_time: int = 5,
        occupancy: int = 4,
        fare: float = 3.0,
    ):
        """
        入口：读取北京 Excel → 得到当日 PEK+PKX 合并的到/离港分钟序列 →
        调用原有调度函数 → 返回 (schedule, pax_arrival_times)
        """

        # 1) 从北京 Excel 解析当日“合并到/离港时序”
        arr_df, dep_df, yearly_capacity = self.sched.get_one_day_beijing(
            month=month, day=day
        )

        # 2) 下游：调用项目原有的调度函数（无需改动 utils/schedule_utils.py）
        from .utils.schedule_utils import generate_uam_schedule, generate_uam_schedule_v2

        if vertiport_pmf.ndim == 1:
            # 1D PMF：需要 directional_demand 作为总量
            schedule, pax_arrival_times = generate_uam_schedule(
                arr_df,
                dep_df,
                yearly_capacity,
                auto_regressive_alpha,
                directional_demand,
                vertiport_pmf,
                occupancy,
                max_waiting_time,
                self.vertiport_dict_inv,
                self.flight_distance_matrix,
                fare,
            )
        elif vertiport_pmf.ndim == 3:
            # 3D PMF：自带时段/方向维度，不需要 directional_demand
            schedule, pax_arrival_times = generate_uam_schedule_v2(
                arr_df,
                dep_df,
                auto_regressive_alpha,
                vertiport_pmf,
                occupancy,
                max_waiting_time,
                self.vertiport_dict_inv,
                self.flight_distance_matrix,
                fare,
            )
        else:
            raise ValueError("vertiport_pmf 维度必须为 1 或 3。")

        self.schedule = schedule
        self.pax_arrival_times = pax_arrival_times
        return schedule, pax_arrival_times

    def plot_flight(self, ylim=(0, 25)):
        schedule = self.schedule.copy()
        schedule["hour"] = schedule["schedule"] // 60
        schedule.loc[schedule["hour"] == 24.0, "hour"] = 0

        flight_count = schedule.groupby(["hour", "od"]).size().reset_index(name="count")
        pivot_table = flight_count.pivot_table(
            index="hour", columns="od", values="count", fill_value=0
        )
        flight_count = pivot_table.reset_index().melt(
            id_vars="hour", var_name="od", value_name="count"
        )

        flight_count["origin"] = flight_count["od"].apply(lambda x: x.split("_")[0])
        flight_count["destination"] = flight_count["od"].apply(
            lambda x: x.split("_")[1]
        )
        flight_count = flight_count.fillna(0)

        input_to_viz = np.zeros((len(self.vertiports), len(self.vertiports), 24))
        for idx, row in flight_count.iterrows():
            o = row["origin"]
            d = row["destination"]
            h = int(row["hour"])
            input_to_viz[self.vertiport_dict[o], self.vertiport_dict[d], h] = row[
                "count"
            ]

        fig, ax = plot_travel_time(
            input_to_viz, self.vertiports, ylim=ylim, ylabel="Number of Flights"
        )

        return fig, ax

    def plot_pax(self, ylim=(0, 80)):
        pax_arrival_times = self.pax_arrival_times.copy()
        pax_arrival_times["hour"] = (
            pax_arrival_times["passenger_arrival_time_s"] // 3600
        )
        pax_count = (
            pax_arrival_times.groupby(
                ["origin_vertiport_id", "destination_vertiport_id", "hour"]
            )
            .size()
            .reset_index(name="counts")
        )

        input_to_viz = np.zeros((len(self.vertiports), len(self.vertiports), 24))
        for idx, row in pax_count.iterrows():
            o = int(row["origin_vertiport_id"])
            d = int(row["destination_vertiport_id"])
            h = int(row["hour"])
            input_to_viz[o, d, h] = row["counts"]

        fig, ax = plot_travel_time(
            input_to_viz, self.vertiports, ylim=ylim, ylabel="Number of Passengers"
        )

        return fig, ax
