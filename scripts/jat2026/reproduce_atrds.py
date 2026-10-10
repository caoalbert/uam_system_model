"""JAT2026 / F2-0: reproduce the ATRDS 2026 LAX results with the ORIGINAL rules.

Faithful script version of Albert's files in external/reference/
(1b-lax-run-script.py, 1a-lax-xp.ipynb cells 4-27 and 30-58, 4-phase2-script.py).
Nothing is "fixed" here on purpose (decision 2026-10-06): the slot -> hour mapping
(slot // 2), alpha 0.7 (stage 1, $3/mi) vs 0.1 (variable-pricing stage 2), the +10 min
hub-spoke flight time in stage 2 only, and Pricing.py as on main are all kept.
A corrected run comes later.

Run from the repo root with the wisk venv, e.g.
    $py = "<path to your venv>\\Scripts\\python.exe"
    & $py scripts\\jat2026\\reproduce_atrds.py all --fleets 30            # Table IV
    & $py scripts\\jat2026\\reproduce_atrds.py stage1 --fleets 20-70      # Fig. 10 (6 x <= 4 h)
Steps: stage1 -> bridge -> stage2 -> analyze ("all" runs them in order and skips
steps whose output already exists unless --force).
Outputs: runs/atrds_repro/{stage1,networks,stage2,analysis}/
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from geopy.distance import geodesic

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from uam_system_model.StarNetwork import StarNetwork  # noqa: E402

DATA = os.path.join(ROOT, "uam_system_model", "data")
CFG = os.path.join(DATA, "config")
OUT = os.path.join(ROOT, "runs", "atrds_repro")

# ---------------------------------------------------------------- settings (1a / 1b / 4)
VERTIPORTS = ["LAX", "DTLA", "LGB", "WDHL", "UVS", "ANH", "HWD", "PSD", "BVH"]
COORDINATES = np.array([
    (33.94417072663332, -118.40248449470265),
    (34.043119432990345, -118.26720629762752),
    (33.81606030845284, -118.15122129260827),
    (34.17160155814583, -118.60525790222589),
    (34.13843197335217, -118.35511311047983),
    (33.812102161009804, -117.91897753216998),
    (34.09174681098575, -118.32725521957076),
    (34.08807589928915, -118.55063573741867),
    (34.06666969690935, -118.4114352069469),
])
SURVEY_P_TNC = 0.33327
OD_PAX = 77_500_000
VERTIPORT_PMF = np.array([0, 3.8, 2.4, 1, 1, 4.1, 2.5, 2.2, 1.5]) / 100
UBER_FARE = np.array([0, 50, 62, 78, 57, 77, 57, 57, 45])
MONTH, DAY, SEED, MAX_WAIT = 9, 5, 9, 10
ALPHA_STAGE1 = 0.7          # 1b, and $3/mi (1a cell 31)
ALPHA_STAGE2_VARIABLE = 0.1  # 1a cell 24
STAGE2_EXTRA_SLOTS = 2       # 1a cell 18: +2 x 5 min on hub<->spoke flights, stage 2 only
S1 = dict(time_resolution=30, optimality_gap=0.05, value_of_time=32.63,
          utility_type="betas", time_limit=3600 * 4)
S2 = dict(max_zero_prep_time=6, max_tour_energy=0.4 * 100, beta_0_coefficient=4, epsilon=0)

# ATRDS Table IV (fleet 30) and Fig. 10 / notebook cell 56 (variable pricing)
REFERENCE = {
    ("fixed3", 30): dict(total_pads=57, number_of_revenue_flights=1071, number_of_repositioning_flights=128,
                         number_of_revenue_passengers=3837, average_load_factor=0.800,
                         number_of_aircraft_hours=402, RAM=12987, TAM=14387, RASM=2.402, CASM=1.503,
                         daily_revenue=138203, daily_opex=86486, daily_profit=51717),
    ("variable", 30): dict(total_pads=59, number_of_revenue_flights=936, number_of_repositioning_flights=106,
                           number_of_revenue_passengers=2797, average_load_factor=0.671,
                           number_of_aircraft_hours=371, RAM=15181, TAM=16648, RASM=3.123, CASM=1.279,
                           daily_revenue=207938, daily_opex=85169, daily_profit=122769, IRR=0.5003),
    ("variable", 20): dict(RASM=3.1066, CASM=1.4534, IRR=0.3875),
    ("variable", 40): dict(RASM=2.9500, CASM=1.1896, IRR=0.5337),
    ("variable", 50): dict(RASM=2.8665, CASM=1.1322, IRR=0.5559, number_of_revenue_flights=1479,
                           number_of_revenue_passengers=4848, daily_revenue=301824),
    ("variable", 60): dict(RASM=2.7398, CASM=1.0860, IRR=0.5398),
    ("variable", 70): dict(RASM=2.6645, CASM=1.0703, IRR=0.5162),
}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def distance_matrix():
    n = len(COORDINATES)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i][j] = geodesic(COORDINATES[i], COORDINATES[j]).meters / 1600
    return d


def build_network(extra_slots=0):
    ft = np.load(os.path.join(CFG, "flight_time_matrix.npy"))
    if extra_slots:
        ft = ft.copy()
        ft[0, 1:] += extra_slots
        ft[1:, 0] += extra_slots
    ec = np.load(os.path.join(CFG, "energy_consumption_matrix.npy"))
    return StarNetwork(vertiport_names=VERTIPORTS, flight_distance_matrix=distance_matrix(),
                       flight_time_matrix=ft, energy_consumption_matrix=ec)


def scale_shares(network, share):
    """share (9,9,24) UAM share -> (2,9,24) arrival-rate array, as in 1a cells 8/21/31."""
    total_seat = network.demand_generator.yearly_capacity[0] + network.demand_generator.yearly_capacity[1]
    density = np.zeros((9, 9))
    density[0, :] = VERTIPORT_PMF
    density[:, 0] = VERTIPORT_PMF
    p = share * SURVEY_P_TNC * (OD_PAX / total_seat)
    p = p * np.repeat(density[:, :, np.newaxis], 24, axis=2)
    return np.concatenate([[p[0, :, :], p[:, 0, :]]])


def parse_fleets(s):
    if "-" in s:
        a, b = (int(x) for x in s.split("-"))
        return list(range(a, b + 1, 10))
    return [int(x) for x in s.split(",")]


# ---------------------------------------------------------------- stage 1 (1b)
def stage1_path(K):
    return os.path.join(OUT, "stage1", f"pricing_{K}.csv")


def run_stage1(K, force=False):
    from uam_system_model.Pricing import PricingOptimizer
    out = stage1_path(K)
    if os.path.exists(out) and not force:
        log(f"stage1 K={K}: exists, skip ({out})")
        return
    ensure(os.path.dirname(out))
    network = build_network()
    network.load_demand(month=MONTH, day=DAY, vertiport_pmf=scale_shares(network, np.ones((9, 9, 24))),
                        max_waiting_time=MAX_WAIT, auto_regressive_alpha=ALPHA_STAGE1, seed=SEED)
    with open(os.path.join(DATA, "2-travel_time.pkl"), "rb") as f:
        uber_tt, first_mile, last_mile, dist_avg = pickle.load(f)
    flm = np.zeros((9, 9, 24))
    flm[0, :, :] = dist_avg
    flm[:, 0, :] = dist_avg
    uam_ft = np.load(os.path.join(CFG, "flight_time_matrix.npy")) * 5
    log(f"stage1 K={K}: solving (gap {S1['optimality_gap']}, limit {S1['time_limit']} s)")
    t0 = time.time()
    df, repo = PricingOptimizer(StarNetwork=network).optimize(
        num_vehicles=K, uber_travel_time=uber_tt, first_mile_time=first_mile, last_mile_time=last_mile,
        first_or_last_distance=flm, uam_flight_time=uam_ft, uam_distance_matrix=distance_matrix(),
        uber_fare=UBER_FARE, **S1)
    df.to_csv(out, index=False)
    repo.to_csv(out.replace(".csv", "_repo.csv"), index=False)
    log(f"stage1 K={K}: done in {(time.time() - t0) / 60:.1f} min -> {out}")


# ---------------------------------------------------------------- bridge (1a cells 16-27, 31)
def network_path(scenario, K):
    return os.path.join(OUT, "networks", f"{scenario}_K{K}.pkl")


def run_bridge_variable(K, stage1_K=None, force=False):
    stage1_K = stage1_K or K
    out = network_path("variable", K)
    if os.path.exists(out) and not force:
        log(f"bridge variable K={K}: exists, skip")
        return
    ensure(os.path.dirname(out))
    df = pd.read_csv(stage1_path(stage1_K))
    df["hour"] = df["passenger_arrival_time_slot"] // 2          # original (see TODO: off by one)
    df.loc[df["hour"] == 24, "hour"] = 23
    itr = (df.groupby(["hour", "origin_vertiport_id", "destination_vertiport_id"])[["fare", "percentage_uam"]]
           .mean().reset_index())
    network = build_network(extra_slots=STAGE2_EXTRA_SLOTS)
    share = np.zeros((9, 9, 24))
    for _, r in itr.iterrows():
        share[int(r["origin_vertiport_id"]), int(r["destination_vertiport_id"]), int(r["hour"])] = r["percentage_uam"]
    network.load_demand(month=MONTH, day=DAY, vertiport_pmf=scale_shares(network, share),
                        max_waiting_time=MAX_WAIT, auto_regressive_alpha=ALPHA_STAGE2_VARIABLE, seed=SEED)
    sched = network.schedule.copy()
    sched["hour"] = sched["schedule"] // 60
    inv = network.vertiport_dict_inv
    itr["od"] = [f"{inv[o]}_{inv[d]}" for o, d in zip(itr["origin_vertiport_id"], itr["destination_vertiport_id"])]
    new = pd.merge(sched, itr, on=["hour", "od"], how="left")
    new["revenue"] = new["num_pax"] * new["fare"]
    new = new[["schedule", "od", "num_pax", "revenue"]]
    fallback = new["revenue"].isna()

    def uber_fallback(row):
        if pd.isna(row["revenue"]):
            o, d = (network.vertiport_dict[x] for x in row["od"].split("_"))
            if o == 0:
                return UBER_FARE[d] * row["num_pax"]
            if d == 0:
                return UBER_FARE[o] * row["num_pax"]
            return None
        return round(row["revenue"], 2)

    new["revenue"] = new.apply(uber_fallback, axis=1)
    network.schedule = new.copy()
    with open(out, "wb") as f:
        pickle.dump(network, f)
    info = dict(scenario="variable", K=K, stage1_K=stage1_K, flights=int(len(new)), pax=int(new["num_pax"].sum()),
                potential_revenue=float(new["revenue"].sum()), uber_fallback_flights=int(fallback.sum()),
                uber_fallback_revenue=float(new.loc[fallback, "revenue"].sum()))
    with open(out.replace(".pkl", "_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    log(f"bridge variable K={K}: {info}")


def run_bridge_fixed3(K, force=False):
    out = network_path("fixed3", K)
    if os.path.exists(out) and not force:
        log(f"bridge fixed3 K={K}: exists, skip")
        return
    ensure(os.path.dirname(out))
    network = build_network(extra_slots=STAGE2_EXTRA_SLOTS)    # cell 31 ran on the +10 min network
    share = np.load(os.path.join(CFG, "p_uam_array_10min_wait_deterministic.npy"))
    network.load_demand(month=MONTH, day=DAY, vertiport_pmf=scale_shares(network, share),
                        max_waiting_time=MAX_WAIT, auto_regressive_alpha=ALPHA_STAGE1)  # seed default 9, fare $3/mi
    with open(out, "wb") as f:
        pickle.dump(network, f)
    info = dict(scenario="fixed3", K=K, flights=int(len(network.schedule)),
                pax=int(network.schedule["num_pax"].sum()), potential_revenue=float(network.schedule["revenue"].sum()))
    with open(out.replace(".pkl", "_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    log(f"bridge fixed3 K={K}: {info}")


# ---------------------------------------------------------------- stage 2 (4-phase2-script.py)
def stage2_path(scenario, K):
    return os.path.join(OUT, "stage2", f"{scenario}_K{K}.csv")


def run_stage2(scenario, K, n_processes, force=False):
    from uam_system_model.FleetOpVRP import FleetOpVRP
    out = stage2_path(scenario, K)
    if os.path.exists(out) and not force:
        log(f"stage2 {scenario} K={K}: exists, skip")
        return
    ensure(os.path.dirname(out))
    with open(network_path(scenario, K), "rb") as f:
        network = pickle.load(f)
    log(f"stage2 {scenario} K={K}: solving with {n_processes} processes")
    t0 = time.time()
    unserved, served = FleetOpVRP(StarNetwork=network).optimize(
        K, S2["max_zero_prep_time"], S2["max_tour_energy"], beta_0_coefficient=S2["beta_0_coefficient"],
        n_processes=n_processes, epsilon=S2["epsilon"], verbose=False)
    served.to_csv(out, index=False)
    unserved.to_csv(out.replace(".csv", "_unserved.csv"), index=False)
    log(f"stage2 {scenario} K={K}: done in {(time.time() - t0) / 60:.1f} min -> {out}")


# ---------------------------------------------------------------- analytics (1a cells 30-58)
def analyze(runs):
    import matplotlib
    matplotlib.use("Agg")
    from uam_system_model.utils.FleetOpVRPSummary import FleetOpVRPSummary
    from uam_system_model.financial_analysis.cost import CostAnalyzer
    rows = []
    for scenario, K in runs:
        if not os.path.exists(stage2_path(scenario, K)):
            log(f"analyze: missing {stage2_path(scenario, K)}, skip")
            continue
        with open(network_path(scenario, K), "rb") as f:
            network = pickle.load(f)
        stats = FleetOpVRPSummary(network, pd.read_csv(stage2_path(scenario, K))).get_summary_statistics()
        c = CostAnalyzer(os.path.join(CFG, "financial_analysis_params.json"))
        capex, opex, rev, prof = c.analyze(pd.DataFrame([stats]))
        _, _, roi, npv, irr, payback = c.projection()
        row = {k: v for k, v in stats.items() if not isinstance(v, dict)}
        row.update(scenario=scenario, K=K, capex=capex, daily_revenue=rev / 365, daily_opex=opex / 365,
                   daily_profit=prof / 365, RASM=rev / stats["TAM"] / 4 / 365, CASM=opex / stats["TAM"] / 4 / 365,
                   revenue_per_aircraft_hour=rev / 365 / stats["number_of_aircraft_hours"],
                   ROI=roi, NPV=npv, IRR=irr, payback_year=payback)
        rows.append(row)
    if not rows:
        return
    res = pd.DataFrame(rows)
    ensure(os.path.join(OUT, "analysis"))
    res.to_csv(os.path.join(OUT, "analysis", "summary.csv"), index=False)
    # side-by-side with the ATRDS numbers
    cmp_rows = []
    for _, r in res.iterrows():
        for metric, ref in REFERENCE.get((r["scenario"], int(r["K"])), {}).items():
            new = r.get(metric)
            cmp_rows.append(dict(scenario=r["scenario"], K=int(r["K"]), metric=metric, atrds=ref, reproduced=new,
                                 diff_pct=None if new is None or ref == 0 else 100 * (new - ref) / ref))
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(os.path.join(OUT, "analysis", "compare_atrds.csv"), index=False)
    if cmp.empty:
        log("analysis: no ATRDS reference values for these runs")
        return
    with pd.option_context("display.max_rows", 200, "display.width", 160, "display.float_format", "{:,.3f}".format):
        print(cmp.to_string(index=False))
    log(f"analysis -> {os.path.join(OUT, 'analysis')}")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["stage1", "bridge", "stage2", "analyze", "all"])
    ap.add_argument("--fleets", default="30", help="'30', '20,30' or range '20-70' (step 10)")
    ap.add_argument("--scenarios", default="variable,fixed3", help="variable and/or fixed3 ($3/mi)")
    ap.add_argument("--stage1-fleet", type=int, default=None,
                    help="stage-1 fleet whose fares feed the bridge (default: same K). The notebook read _30.csv.")
    ap.add_argument("--n-processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--force", action="store_true", help="recompute even if output exists")
    a = ap.parse_args()
    fleets = parse_fleets(a.fleets)
    scenarios = [s.strip() for s in a.scenarios.split(",")]
    runs = [(s, K) for s in scenarios for K in fleets]

    if a.step in ("stage1", "all") and "variable" in scenarios:
        for K in fleets:
            run_stage1(K, a.force)
    if a.step in ("bridge", "all"):
        for s, K in runs:
            run_bridge_variable(K, a.stage1_fleet, a.force) if s == "variable" else run_bridge_fixed3(K, a.force)
    if a.step in ("stage2", "all"):
        for s, K in runs:
            run_stage2(s, K, a.n_processes, a.force)
    if a.step in ("analyze", "all"):
        analyze(runs)


if __name__ == "__main__":
    main()
