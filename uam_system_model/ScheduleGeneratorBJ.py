from pathlib import Path

import pandas as pd


class ScheduleGenerator:
    """
    北京专用：读取包含四个 sheet 的 Excel：
        - "PKX Arrivals"     (含 Date, Time)
        - "PKX Departures"   (含 Date, Time)
        - "PEK Arrivals"     (含 Scheduled Time)
        - "PEK Departures"   (含 Scheduled Time)
    """

    def __init__(self, path_schedule: str):
        self.path = Path(path_schedule)

    # ------------------------------------------------------------------
    #  工具函数：把时间列转换为“当天分钟数”
    # ------------------------------------------------------------------
    @staticmethod
    def _to_min_of_day_from_datetime(df: pd.DataFrame, date_col: str, time_col: str):
        dt = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str))
        return dt.dt.hour * 60 + dt.dt.minute

    @staticmethod
    def _to_min_of_day_from_timeonly(df: pd.DataFrame, time_col: str):
        t = pd.to_datetime(df[time_col].astype(str))
        return t.dt.hour * 60 + t.dt.minute

    # ------------------------------------------------------------------
    #  对单个 hub 抽取某一天的到离港时刻（分钟）+ capacity
    # ------------------------------------------------------------------
    def get_one_day_for_hub(self, month: int, day: int, hub_name: str):
        hub_name = hub_name.upper()
        path = self.path

        if hub_name == "PKX":
            # --- PKX Arrivals ---
            pkx_arr = pd.read_excel(path, sheet_name="PKX Arrivals")
            if {"Date", "Time"}.issubset(pkx_arr.columns):
                dt = pd.to_datetime(pkx_arr["Date"])
                mask = (dt.dt.month == month) & (dt.dt.day == day)
                pkx_arr_m = self._to_min_of_day_from_datetime(
                    pkx_arr.loc[mask], "Date", "Time"
                )
            else:
                pkx_arr_m = pd.Series(dtype=int)

            # --- PKX Departures ---
            pkx_dep = pd.read_excel(path, sheet_name="PKX Departures")
            if {"Date", "Time"}.issubset(pkx_dep.columns):
                dt = pd.to_datetime(pkx_dep["Date"])
                mask = (dt.dt.month == month) & (dt.dt.day == day)
                pkx_dep_m = self._to_min_of_day_from_datetime(
                    pkx_dep.loc[mask], "Date", "Time"
                )
            else:
                pkx_dep_m = pd.Series(dtype=int)

            arr_times = pkx_arr_m
            dep_times = pkx_dep_m

        elif hub_name == "PEK":
            # --- PEK Arrivals ---
            pek_arr = pd.read_excel(path, sheet_name="PEK Arrivals")
            if {"Scheduled Time"}.issubset(pek_arr.columns):
                pek_arr_m = self._to_min_of_day_from_timeonly(pek_arr, "Scheduled Time")
            else:
                pek_arr_m = pd.Series(dtype=int)

            # --- PEK Departures ---
            pek_dep = pd.read_excel(path, sheet_name="PEK Departures")
            if {"Scheduled Time"}.issubset(pek_dep.columns):
                pek_dep_m = self._to_min_of_day_from_timeonly(pek_dep, "Scheduled Time")
            else:
                pek_dep_m = pd.Series(dtype=int)

            arr_times = pek_arr_m
            dep_times = pek_dep_m

        else:
            raise ValueError("hub_name 必须为 'PEK' 或 'PKX'")

        # 生成 DataFrame，并加上 capacity 列
        arr_df = pd.DataFrame({"time": arr_times.sort_values().to_numpy(dtype=int)})
        dep_df = pd.DataFrame({"time": dep_times.sort_values().to_numpy(dtype=int)})

        SEATS_PER_FLIGHT = 150
        arr_df["capacity"] = SEATS_PER_FLIGHT
        dep_df["capacity"] = SEATS_PER_FLIGHT

        # 假设Excel表只包含一天的航班，则年容量为每天航班数乘以座位数再乘以365
        yearly_capacity = (
            len(arr_df) * SEATS_PER_FLIGHT * 365,
            len(dep_df) * SEATS_PER_FLIGHT * 365,
        )

        return arr_df, dep_df, yearly_capacity
