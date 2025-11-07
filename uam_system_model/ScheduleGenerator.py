import pandas as pd
from pathlib import Path


class ScheduleGenerator:
    """
    北京专用：读取包含四个 sheet 的 Excel（与原 LAX_ind.csv 文件“放置位置相同”）：
        - 'PKX Arrivals'     (含 Date, Time)
        - 'PKX Departures'   (含 Date, Time)
        - 'PEK Arrivals'     (含 Scheduled Time)
        - 'PEK Departures'   (含 Scheduled Time)

    输出：
        - arr_df: 两场合并的“到港分钟序列”DataFrame，仅一列 'time'（0–1440）
        - dep_df: 两场合并的“离港分钟序列”DataFrame，仅一列 'time'
        - yearly_capacity:  (len(arr_df)*150, len(dep_df)*150)，没有具体机队信息，150 为假设平均座位数
    """

    def __init__(self, path_schedule: str):
        self.path = Path(path_schedule)
        if not self.path.exists():
            raise FileNotFoundError(f"北京数据文件不存在：{self.path}")

        # 备查：必须包含四个 sheet
        xls = pd.ExcelFile(self.path)
        required_sheets = [
            "PKX Arrivals",
            "PKX Departures",
            "PEK Arrivals",
            "PEK Departures",
        ]
        missing = [s for s in required_sheets if s not in xls.sheet_names]
        if missing:
            raise ValueError(f"Excel 缺少必要 sheet：{missing}")

    # ---------- helpers ----------
    @staticmethod
    def _to_min_of_day_from_datetime(df: pd.DataFrame, col_date: str, col_time: str) -> pd.Series:
        """将 'YYYY-MM-DD' + 'HH:MM:SS' 合成为当天分钟数（0-1439）。"""
        t = pd.to_datetime(df[col_date].astype(str) + " " + df[col_time].astype(str))
        return (t.dt.hour * 60 + t.dt.minute).astype(int)

    @staticmethod
    def _to_min_of_day_from_timeonly(df: pd.DataFrame, col_time: str) -> pd.Series:
        """仅有时间列（无日期）时，转换为当天分钟数（0-1439）。"""
        t = pd.to_datetime(df[col_time].astype(str))
        return (t.dt.hour * 60 + t.dt.minute).astype(int)

    # ---------- main ----------
    def get_one_day_beijing(self, month: int, day: int):
        """
        month 用作标签（不做筛选）；按 'Date' 的“日”筛 PKX 两个表；
        PEK 两个表只有时刻，直接按“当天时刻”处理。
        """
        # --- PKX Arrivals ---
        pkx_arr = pd.read_excel(self.path, sheet_name="PKX Arrivals")
        if {"Date", "Time"}.issubset(pkx_arr.columns):
            # 按 'day' 过滤
            pkx_arr_mask = pd.to_datetime(pkx_arr["Date"]).dt.day == day
            pkx_arr = pkx_arr.loc[pkx_arr_mask].copy()
            pkx_arr_m = self._to_min_of_day_from_datetime(pkx_arr, "Date", "Time")
        else:
            pkx_arr_m = pd.Series(dtype=int)

        # --- PKX Departures ---
        pkx_dep = pd.read_excel(self.path, sheet_name="PKX Departures")
        if {"Date", "Time"}.issubset(pkx_dep.columns):
            pkx_dep_mask = pd.to_datetime(pkx_dep["Date"]).dt.day == day
            pkx_dep = pkx_dep.loc[pkx_dep_mask].copy()
            pkx_dep_m = self._to_min_of_day_from_datetime(pkx_dep, "Date", "Time")
        else:
            pkx_dep_m = pd.Series(dtype=int)

        # --- PEK Arrivals (仅时间) ---
        pek_arr = pd.read_excel(self.path, sheet_name="PEK Arrivals")
        # 注意：到达表头为 "Scheduled Time"、"Flight No."、"Airlines"、"Terminal"、"Destination"
        if "Scheduled Time" in pek_arr.columns:
            pek_arr_m = self._to_min_of_day_from_timeonly(pek_arr, "Scheduled Time")
        else:
            pek_arr_m = pd.Series(dtype=int)

        # --- PEK Departures (仅时间) ---
        pek_dep = pd.read_excel(self.path, sheet_name="PEK Departures")
        # 注意：出发表头为 "Scheduled Time"、"Flight number"、"Airlines"、"Terminal"、"Destination"
        if "Scheduled Time" in pek_dep.columns:
            pek_dep_m = self._to_min_of_day_from_timeonly(pek_dep, "Scheduled Time")
        else:
            pek_dep_m = pd.Series(dtype=int)

        # --- 合并为 arr/dep 序列 ---
        arr_times = pd.concat([pkx_arr_m, pek_arr_m], ignore_index=True)
        dep_times = pd.concat([pkx_dep_m, pek_dep_m], ignore_index=True)

        arr_df = pd.DataFrame({"time": arr_times.sort_values().to_numpy(dtype=int)})
        dep_df = pd.DataFrame({"time": dep_times.sort_values().to_numpy(dtype=int)})

        # 年容量假设值
        yearly_capacity = (len(arr_df) * 150, len(dep_df) * 150)

        return arr_df, dep_df, yearly_capacity
