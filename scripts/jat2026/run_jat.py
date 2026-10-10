"""JAT2026: corrected runs + stage-2 -> stage-1 capacity feedback (F2-0 corrected, F2-7).

Corrected settings (decided 2026-10-07/08, see project notes):
  * demand alpha 0.7 in BOTH stages; stage-1 pricing day = 5 Sep 2019 schedule, seed 9
  * +10 min (2 x 5 min) hub<->spoke turnaround in BOTH stages (aircraft side only; the
    passenger travel time in the logit is unchanged and already has a 10-min transition)
  * repositioning time counted once in stage 1 (Pricing repo_time_factor=1)
  * first/last-mile ground cost removed from stage-1 revenue (ground_cost_in_objective=True)
  * bridge: hourly UAM shares with hour = (slot-1)//2; stage-2 fares matched by 30-min slot
Feedback = Algorithm 1 in the paper (Sec. III-E, decided with Andrew 2026-10-08):
  stage 2 measures rho_h = (revenue + repositioning flights departing in hour h) / K;
  hours where stage 2 left revenue flights unserved get a network-wide cap in stage 1:
      sum_{markets in h} x_i + sum_{repositioning legs in h} y_ij <= rho_h * K
  rho_h is averaged over iterations (MSA, lambda_n = 1/(n+1)); new capped hours start at rho_hat.
  Stop when (n >= 1) rho, the fares of markets served in both iterations, and the planned
  flights per hour each change by < 2%.
Iteration 0 = open loop (no caps) = the corrected open-loop result.

Usage (repo root, wisk venv):
  & $py scripts\\jat2026\\run_jat.py converge --fleet 30 --max-iter 5
  & $py scripts\\jat2026\\run_jat.py evaluate --fleet 30 --plan final --seeds 1-5
  & $py scripts\\jat2026\\run_jat.py converge --toy                      # small test, minutes
Outputs: runs/jat/variable_K30/iterN/..., convergence.csv, evaluation_<plan>.csv
Re-running skips steps whose outputs exist (safe to restart after a crash).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reproduce_atrds as R  # noqa: E402  (shared settings, network builder, paths)

ALPHA = 0.7
EXTRA_SLOTS = 2
PRICING_SEED = 9
BRIDGE = "hourly"  # "hourly" (ATRDS: hourly share array -> generator) or "thin" (per-passenger thinning by own-slot share)
SHARE_OK = 0.95
COST_BASIS = "all"  # cost per seat-mile fed back to stage 1 (market method): "all" operating costs or "variable" only
FIXED_COST_PER_FLIGHT = 20  # Pricing.optimize default fixed_cost_per_flight; used only before any stage-2 cost feedback
OUT = os.path.join(R.ROOT, "runs", "jat")
log = R.log
TOY = dict(on=False, n_vertiports=9, fixed_scale=1.0)


def enable_toy(hours=(7, 11), vertiports=("LAX", "DTLA", "ANH")):
    """Small test instance: only the given hours and vertiports (same code path, minutes to run).
    Filters passengers (stage 1) and candidate flights (stage 2) right after demand generation."""
    global OUT
    from uam_system_model.StarNetwork import StarNetwork
    original = StarNetwork.load_demand
    h0, h1 = hours
    keep_ids = [R.VERTIPORTS.index(v) for v in vertiports]
    keep_ods = {f"{a}_{b}" for a in vertiports for b in vertiports if a != b}

    def load_demand_small(self, *args, **kwargs):
        original(self, *args, **kwargs)
        pa = self.pax_arrival_times
        t = pa["passenger_arrival_time_s"]
        self.pax_arrival_times = pa[(t >= h0 * 3600) & (t < h1 * 3600)
                                    & pa["origin_vertiport_id"].isin(keep_ids)
                                    & pa["destination_vertiport_id"].isin(keep_ids)].copy()
        sc = self.schedule
        self.schedule = sc[(sc["schedule"] >= h0 * 60) & (sc["schedule"] < h1 * 60)
                           & sc["od"].isin(keep_ods)].reset_index(drop=True)

    StarNetwork.load_demand = load_demand_small
    TOY["on"] = True
    TOY["n_vertiports"] = len(vertiports)
    TOY["hours"], TOY["original_load"], TOY["fixed_scale_set"] = (h0, h1), original, False
    OUT = os.path.join(R.ROOT, "runs", f"jat_toy_{h0}-{h1}_{'-'.join(vertiports)}")
    log(f"TOY MODE: hours {h0}-{h1}, vertiports {vertiports}, output {OUT}")


# ---------------------------------------------------------------- stage 1
def run_stage1(K, hourly_caps, out_csv, market_caps=None, opex_per_asm=0.79, fixed_cost_per_flight=FIXED_COST_PER_FLIGHT):
    from uam_system_model.Pricing import PricingOptimizer
    network = R.build_network(extra_slots=EXTRA_SLOTS)
    network.load_demand(month=R.MONTH, day=R.DAY, vertiport_pmf=R.scale_shares(network, np.ones((9, 9, 24))),
                        max_waiting_time=R.MAX_WAIT, auto_regressive_alpha=ALPHA, seed=PRICING_SEED)
    with open(os.path.join(R.DATA, "2-travel_time.pkl"), "rb") as f:
        uber_tt, first_mile, last_mile, dist_avg = pickle.load(f)
    flm = np.zeros((9, 9, 24))
    flm[0, :, :] = dist_avg
    flm[:, 0, :] = dist_avg
    uam_ft = np.load(os.path.join(R.CFG, "flight_time_matrix.npy")) * 5   # passenger in-vehicle time (unchanged)
    op = PricingOptimizer(StarNetwork=network)
    S1 = R.S1  # JAT2026: toy uses exactly the full-run stage-1 settings (5% gap, 4 h limit)
    t0 = time.time()
    df, repo = op.optimize(num_vehicles=K, uber_travel_time=uber_tt, first_mile_time=first_mile,
                           last_mile_time=last_mile, first_or_last_distance=flm, uam_flight_time=uam_ft,
                           uam_distance_matrix=R.distance_matrix(), uber_fare=R.UBER_FARE,
                           repo_time_factor=1, ground_cost_in_objective=True, hourly_caps=hourly_caps,
                           market_hour_caps=market_caps, opex_per_asm=opex_per_asm,
                           fixed_cost_per_flight=fixed_cost_per_flight, **S1)
    if op.model.SolCount == 0:  # JAT2026: fail loudly (e.g. caps leave no feasible fleet plan)
        raise RuntimeError(f"stage 1 found no solution (Gurobi status {op.model.Status})")
    df.to_csv(out_csv, index=False)
    repo.to_csv(out_csv.replace(".csv", "_repo.csv"), index=False)
    # all markets with demand, incl. those stage 1 chose not to serve (theta ~ 0), for the bridge
    op.pax_arr_grouped.rename(columns={"counts": "num_pax"}).to_csv(out_csv.replace(".csv", "_markets.csv"), index=False)
    info = dict(objective=float(op.model.ObjVal), mip_gap=float(op.model.MIPGap),
                minutes=(time.time() - t0) / 60, opex_per_asm=float(opex_per_asm),
                fixed_cost_per_flight=float(fixed_cost_per_flight),
                # stage-1 revenue + repositioning flights in each capped hour (same terms as the hourly cap)
                flights_in_capped_hours={str(h): float(sum(v.X for v in xs))
                                         for h, xs in sorted(getattr(op, "hourly_cap_vars", {}).items())})
    with open(out_csv.replace(".csv", "_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return df


# ---------------------------------------------------------------- bridge (corrected)
def load_markets(stage1_csv):
    p = stage1_csv.replace(".csv", "_markets.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def bridge(df1, seed, out_pkl, markets=None):
    df = df1.copy()
    df["hour"] = ((df["passenger_arrival_time_slot"] - 1) // 2).clip(0, 23)
    keys = ["hour", "origin_vertiport_id", "destination_vertiport_id"]
    if markets is not None:
        # JAT2026: hourly UAM share = UAM passengers / market demand over BOTH slots of the hour,
        # so a slot stage 1 chose not to serve counts as share 0 (ATRDS averaged only listed slots).
        mk = markets.copy()
        mk["hour"] = ((mk["passenger_arrival_time_slot"] - 1) // 2).clip(0, 23)
        share_h = (df.groupby(keys)["uam_pax"].sum() / mk.groupby(keys)["num_pax"].sum()).dropna()
    else:
        share_h = df.groupby(keys)["percentage_uam"].mean()
    network = R.build_network(extra_slots=EXTRA_SLOTS)
    if BRIDGE == "thin":
        # JAT2026: the generator indexes rates by the AIRLINE flight hour, while stage-1 markets are indexed by the
        # passenger's arrival hour at the vertiport (~2 h earlier for trips to LAX, ~0.5 h later from LAX), so the
        # hourly share array is applied to the wrong passengers. Instead: draw the full market day (share 1, the
        # same process stage 1 prices), then each passenger takes UAM with the share theta_i of HER OWN market
        # (30-min slot, o, d); thinning a Poisson process by theta_i keeps it Poisson. Rebuild flights (dispatch rule).
        from uam_system_model.utils.schedule_utils import build_schedules
        network.load_demand(month=R.MONTH, day=R.DAY, vertiport_pmf=R.scale_shares(network, np.ones((9, 9, 24))),
                            max_waiting_time=R.MAX_WAIT, auto_regressive_alpha=ALPHA, seed=seed)
        theta = df.groupby(["passenger_arrival_time_slot", "origin_vertiport_id", "destination_vertiport_id"])["percentage_uam"].mean()
        pa = network.pax_arrival_times.copy()
        slot = np.ceil(pa["passenger_arrival_time_s"] / 1800).clip(1, 48).astype(int)
        th = np.array([theta.get((s_, o, d), 0.0) for s_, o, d in
                       zip(slot, pa["origin_vertiport_id"], pa["destination_vertiport_id"])])
        fare_m = df.groupby(["passenger_arrival_time_slot", "origin_vertiport_id", "destination_vertiport_id"])["fare"].mean()
        rng = np.random.default_rng(1000 + seed)
        keep = rng.random(len(pa)) < th
        pa = pa[keep].reset_index(drop=True)
        pa["fare"] = [fare_m[(s_, o, d)] for s_, o, d in zip(slot[keep], pa["origin_vertiport_id"], pa["destination_vertiport_id"])]
        inv0 = network.vertiport_dict_inv
        rows = []
        for (o, d), g in pa.groupby(["origin_vertiport_id", "destination_vertiport_id"]):
            g = g.sort_values("passenger_arrival_time_s")
            t, n = build_schedules(g["passenger_arrival_time_s"].values / 60, R.MAX_WAIT, 4)
            # dispatch rule fills flights with consecutive passengers -> revenue = sum of each passenger's own fare
            edges = np.concatenate([[0], np.cumsum(n)]).astype(int)
            cf = np.concatenate([[0.0], np.cumsum(g["fare"].values)])
            rev = cf[edges[1:]] - cf[edges[:-1]]
            rows.append(pd.DataFrame({"schedule": t, "od": f"{inv0[int(o)]}_{inv0[int(d)]}", "num_pax": n, "revenue": rev.round(2)}))
        network.pax_arrival_times = pa
        network.schedule = (pd.concat(rows, ignore_index=True) if rows else
                            pd.DataFrame(columns=["schedule", "od", "num_pax", "revenue"])).sort_values(["schedule", "od"]).reset_index(drop=True)
    else:
        share = np.zeros((9, 9, 24))
        for (h, o, d), v in share_h.items():
            share[int(o), int(d), int(h)] = v
        network.load_demand(month=R.MONTH, day=R.DAY, vertiport_pmf=R.scale_shares(network, share),
                            max_waiting_time=R.MAX_WAIT, auto_regressive_alpha=ALPHA, seed=seed)
    if BRIDGE == "thin":
        sched = network.schedule
        with open(out_pkl, "wb") as f:
            pickle.dump(network, f)
        return network, dict(flights=int(len(sched)), pax=int(sched["num_pax"].sum()),
                             potential_revenue=float(sched["revenue"].sum()), fare="own market (slot, o, d)")
    inv = network.vertiport_dict_inv
    df["od"] = [f"{inv[o]}_{inv[d]}" for o, d in zip(df["origin_vertiport_id"], df["destination_vertiport_id"])]
    fare_slot = df.groupby(["passenger_arrival_time_slot", "od"])["fare"].mean()
    fare_hour = df.groupby(["hour", "od"])["fare"].mean()
    sched = network.schedule.copy()
    sched["slot"] = np.ceil(sched["schedule"] / 30).clip(1, 48).astype(int)
    sched["hour"] = (sched["schedule"] // 60).clip(0, 23).astype(int)
    fares, src = [], []
    for slot, hour, od in zip(sched["slot"], sched["hour"], sched["od"]):
        if (slot, od) in fare_slot.index:
            fares.append(fare_slot[(slot, od)]); src.append("slot")
        elif (hour, od) in fare_hour.index:
            fares.append(fare_hour[(hour, od)]); src.append("hour")
        else:
            o, d = (network.vertiport_dict[x] for x in od.split("_"))
            fares.append(float(R.UBER_FARE[max(o, d)])); src.append("uber")
    sched["revenue"] = (sched["num_pax"] * np.array(fares)).round(2)
    network.schedule = sched[["schedule", "od", "num_pax", "revenue"]].copy()
    with open(out_pkl, "wb") as f:
        pickle.dump(network, f)
    src = pd.Series(src)
    return network, dict(flights=int(len(sched)), pax=int(sched["num_pax"].sum()),
                         potential_revenue=float(sched["revenue"].sum()),
                         fare_by_slot=int((src == "slot").sum()), fare_by_hour=int((src == "hour").sum()),
                         fare_uber=int((src == "uber").sum()))


# ---------------------------------------------------------------- stage 2 + metrics
def run_stage2(network, K, n_processes, out_csv):
    from uam_system_model.FleetOpVRP import FleetOpVRP
    if len(network.schedule) < K:
        raise RuntimeError(f"stage 2 needs at least K={K} candidate flights (every aircraft must fly); "
                           f"got {len(network.schedule)}. Use a smaller --fleet or more --toy-hours.")
    t0 = time.time()
    unserved, served = FleetOpVRP(StarNetwork=network).optimize(
        K, R.S2["max_zero_prep_time"], R.S2["max_tour_energy"], beta_0_coefficient=R.S2["beta_0_coefficient"],
        n_processes=n_processes, epsilon=R.S2["epsilon"], verbose=False)
    served.to_csv(out_csv, index=False)
    unserved.to_csv(out_csv.replace(".csv", "_unserved.csv"), index=False)
    log(f"  stage 2 done in {(time.time() - t0) / 60:.1f} min")
    return served


def realized_metrics(network, served):
    import matplotlib
    matplotlib.use("Agg")
    from uam_system_model.utils.FleetOpVRPSummary import FleetOpVRPSummary
    from uam_system_model.financial_analysis.cost import CostAnalyzer
    stats = FleetOpVRPSummary(network, served).get_summary_statistics()
    c = CostAnalyzer(os.path.join(R.CFG, "financial_analysis_params.json"))
    capex, opex, rev, prof = c.analyze(pd.DataFrame([stats]))
    row = {k: v for k, v in stats.items() if not isinstance(v, dict)}
    row.update(daily_revenue=rev / 365, daily_opex=opex / 365, daily_profit=prof / 365,
               RASM=rev / stats["TAM"] / 4 / 365, CASM=opex / stats["TAM"] / 4 / 365, capex=capex)
    return row


def run_dir_name(method, K, step_up=2.0):
    """Folder of a feedback run (shared by converge, converge_market and evaluate)."""
    if method == "market":
        return f"market_K{K}_v2" + ("_variablecost" if COST_BASIS == "variable" else "")
    return f"hourly_cost_K{K}" + ("" if step_up == 2 else f"_step{step_up:g}")


def operating_cost(stats, K):
    """Daily operating cost of the stage-2 schedule, same formulas as cost.py (CostAnalyzer._compute_opex):
    variable (energy, pilot, battery replacement, maintenance) + fixed (insurance, vertiport operation),
    grossed up by the administrative factor. Toy runs: fixed costs only for the toy vertiports and scaled by the
    share of the day's travelers inside the toy window, so cost per seat-mile is comparable to the full day.
    Returns (daily opex, cost per seat-mile for stage 1 = opex / (4 x aircraft miles))."""
    if TOY["on"] and not TOY["fixed_scale_set"]:
        # share of the pricing day's travelers (all vertiports, all routes) whose vertiport arrival is inside the window
        net = R.build_network(extra_slots=EXTRA_SLOTS)
        TOY["original_load"](net, month=R.MONTH, day=R.DAY, vertiport_pmf=R.scale_shares(net, np.ones((9, 9, 24))),
                             max_waiting_time=R.MAX_WAIT, auto_regressive_alpha=ALPHA, seed=PRICING_SEED)
        t = net.pax_arrival_times["passenger_arrival_time_s"]
        h0, h1 = TOY["hours"]
        TOY["fixed_scale"], TOY["fixed_scale_set"] = float(((t >= h0 * 3600) & (t < h1 * 3600)).mean()), True
        log(f"  toy fixed-cost scale (share of the day's travelers inside {h0}-{h1} h): {TOY['fixed_scale']:.3f}")
    with open(os.path.join(R.CFG, "financial_analysis_params.json")) as f:
        P = json.load(f)
    o, c = P["opex"], P["capex"]
    var = (o["energy_cost_per_kWh"] * stats["energy_consumption_kWh"]
           + (o["pilot_cost_per_aircraft_hour"] + o["battery_replacement_cost_per_aircraft_hour"]) * stats["number_of_aircraft_hours"]
           + o["maintenance_cost_per_asm"] * stats["TAM"] * 4)
    fixed = (K * c["cost_per_vehicle"] * o["insurance_cost_factor"]
             + o["vertiport_operation_cost_per_year"] * TOY["n_vertiports"]) / 365 * TOY["fixed_scale"]
    opex = (var + fixed) / (1 - o["administrative_cost_factor"])
    if COST_BASIS == "variable":  # only costs that change with flying (energy, pilot, battery, maintenance)
        opex_rate = var
    else:  # "all": everything in cost.py (Andrew, 2026-10-09)
        opex_rate = opex
    asm = 4 * stats["TAM"]
    # JAT2026 (Andrew, 2026-10-09): stage 2's cost goes back as a pure cost per seat-mile; stage 1 then drops its $20/flight
    rate = max(0.0, opex_rate / asm) if asm > 0 else float("nan")
    return opex, rate


def served_share_by_hour(network, served):
    """Share of stage-2 candidate flights served, per (origin, destination, hour)."""
    vd = network.vertiport_dict
    cand = {}
    for t, od in zip(network.schedule["schedule"], network.schedule["od"]):
        o, d = (vd[x] for x in od.split("_"))
        k = (o, d, int(min(t // 60, 23)))
        cand[k] = cand.get(k, 0) + 1
    ft_min = np.asarray(network.flight_time) * 5
    got = {}
    for t, seq in zip(served["time"], served["tour_sequence"]):
        legs = [vd[x] for x in str(seq).split("-")]
        for a, b in zip(legs[:-1], legs[1:]):
            k = (a, b, int(min(t // 60, 23)))
            got[k] = got.get(k, 0) + 1
            t += ft_min[a, b]
    return {k: min(1.0, got.get(k, 0) / n) for k, n in cand.items()}, sum(cand.values()), sum(got.values())


def fare_change(df_new, df_old):
    key = ["origin_vertiport_id", "destination_vertiport_id", "passenger_arrival_time_slot"]
    m = df_new.merge(df_old[key + ["fare"]], on=key, suffixes=("", "_old"))
    w = m["uam_pax"].clip(lower=1e-6)
    return float((w * (m["fare"] - m["fare_old"]).abs() / m["fare_old"].abs()).sum() / w.sum())


def measure_hourly(network, served, K):
    """Algorithm 1, Eq. (rho_hat): per hour h, candidate revenue flights, served revenue flights,
    repositioning flights (departing when the previous tour of the same aircraft ends),
    rho_hat = (served revenue + repositioning) / K, and whether revenue flights were left unserved."""
    vd = network.vertiport_dict
    ft_min = np.asarray(network.flight_time) * 5
    hour = lambda t: int(min(max(t, 0) // 60, 23))
    cand, rev, repo = {}, {}, {}
    for t in network.schedule["schedule"]:
        cand[hour(t)] = cand.get(hour(t), 0) + 1
    for t, seq in zip(served["time"], served["tour_sequence"]):
        legs = [vd[x] for x in str(seq).split("-")]
        for a_, b_ in zip(legs[:-1], legs[1:]):
            rev[hour(t)] = rev.get(hour(t), 0) + 1
            t += ft_min[a_, b_]
    for _, g in served.sort_values("time").groupby("cluster_id"):
        prev = None
        for t, dur, seq in zip(g["time"], g["duration"], g["tour_sequence"]):
            legs = str(seq).split("-")
            if prev is not None and prev[1] != legs[0]:
                repo[hour(prev[0])] = repo.get(hour(prev[0]), 0) + 1
            prev = (t + dur, legs[-1])
    busy = {}
    for t, dur in zip(served["time"], served["duration"]):
        t0, t1 = float(t), float(t) + float(dur)
        while t0 < t1:
            h = hour(t0)
            nxt = min(t1, (int(t0 // 60) + 1) * 60)
            busy[h] = busy.get(h, 0.0) + (nxt - t0)
            t0 = nxt
    out = {}
    for h in range(24):
        c, r, p = cand.get(h, 0), rev.get(h, 0), repo.get(h, 0)
        out[h] = dict(candidates=c, served=r, repositioning=p, rho_hat=(r + p) / K, unserved=c > r,
                      tour_time_share=busy.get(h, 0.0) / (K * 60))  # diagnostic: share of fleet time in revenue tours
    return out


def update_rho(meas, rho_prev, n, K, step_up=2.0, share_ok=SHARE_OK):
    """Algorithm 1 (revised 2026-10-10) cap update, per hour h, rho = flights per aircraft (cap = rho * K):
      share_h = served revenue flights / candidate flights (1 if no candidates)
      share_h <  0.95 -> target = rho_hat_h (stage-2 flights incl. repositioning / K)
      share_h >= 0.95 and h already capped -> target = (max(cap_h, stage-2 flights_h) + step_up) / K   (cap rises)
      new cap = target (first time short); existing cap moves by MSA: rho += (target - rho) / (n + 1)."""
    rho = dict(rho_prev)
    lam = 1.0 / (n + 1)
    for h, m in meas.items():
        share = min(1.0, m["served"] / m["candidates"]) if m["candidates"] > 0 else 1.0
        if share < share_ok:
            target = m["rho_hat"]
        elif h in rho_prev:
            target = (max(rho_prev[h] * K, m["rho_hat"] * K) + step_up) / K
        else:
            continue
        rho[h] = target if h not in rho_prev else rho_prev[h] + lam * (target - rho_prev[h])
    return rho


def rho_change(rho_new, rho_old):
    if not rho_new:
        return 0.0
    if any(h not in rho_old for h in rho_new):
        return float("inf")
    return max(abs(rho_new[h] - rho_old[h]) / max(rho_old[h], 1e-9) for h in rho_new)


def plan_change(df_new, df_old):
    """Relative change of planned flights per hour (stage 1), sum |diff| / total."""
    f = lambda d: d.assign(hour=(d["passenger_arrival_time_slot"] - 1) // 2).groupby("hour")["num_flights"].sum()
    a, b = f(df_new), f(df_old)
    a, b = a.reindex(a.index.union(b.index), fill_value=0), b.reindex(a.index.union(b.index), fill_value=0)
    return float((a - b).abs().sum() / max(b.sum(), 1e-9))


def converge(K, max_iter, n_processes, tol_fare=0.02, tol_plan=0.02, tol_cost=0.02, step_up=2.0):
    """Algorithm 1, revised 2026-10-10 (Andrew): network-wide hourly flight caps + cost feedback.
    Goes back into stage 1 each round:
      (1) per hour h, a cap on all flights (revenue + repositioning between different vertiports), rho_h * K,
          cut toward what stage 2 flew when it left flights unserved, raised by step_up when it served >= 95%;
      (2) stage 2's operating cost per seat-mile (all costs in cost.py / (4 x miles flown)). Round 0 uses
          $0.79/seat-mile + $20/flight (ATRDS); from round 1 the fed-back rate and no $20 per flight.
    Stop (n >= 1): cost gap <= 2% (target), fare change <= 2%, stage-1 flights-per-hour change <= 2%.
    Not converged -> the last round. Diagnostics per round: pax flown / sold, stage-1 vs stage-2 flights per capped hour."""
    base = R.ensure(os.path.join(OUT, run_dir_name("hourly", K, step_up)))
    rho, rate, fixed, rows, df_prev = {}, 0.79, FIXED_COST_PER_FLIGHT, [], None
    for n in range(max_iter):
        d = R.ensure(os.path.join(base, f"iter{n}"))
        caps = {h: r * K for h, r in rho.items()} or None
        with open(os.path.join(d, "inputs.json"), "w") as f:  # what stage 1 is given this round
            json.dump(dict(opex_per_asm=rate, fixed_cost_per_flight=fixed,
                           hourly_caps={str(h): c for h, c in sorted((caps or {}).items())}), f, indent=1)
        log(f"===== K={K} Algorithm 1 (revised), iteration {n}: {len(caps or {})} capped hours, "
            f"cost ${rate:.3f}/seat-mile + ${fixed:.0f}/flight")
        if caps:
            log("  hourly caps (flights): " + ", ".join(f"{h}h:{c:.1f}" for h, c in sorted(caps.items())))
        s1 = os.path.join(d, "stage1.csv")
        df1 = pd.read_csv(s1) if os.path.exists(s1) else run_stage1(K, caps, s1, opex_per_asm=rate,
                                                                     fixed_cost_per_flight=fixed)
        pkl, s2 = os.path.join(d, "network.pkl"), os.path.join(d, "stage2.csv")
        if os.path.exists(pkl) and os.path.exists(s2):
            with open(pkl, "rb") as f:
                network = pickle.load(f)
            served = pd.read_csv(s2)
            with open(os.path.join(d, "bridge_info.json")) as f:
                binfo = json.load(f)
        else:
            network, binfo = bridge(df1, PRICING_SEED, pkl, load_markets(s1))
            with open(os.path.join(d, "bridge_info.json"), "w") as f:
                json.dump(binfo, f, indent=2)
            log(f"  bridge: {binfo}")
            served = run_stage2(network, K, n_processes, s2)
        meas = measure_hourly(network, served, K)
        pd.DataFrame([dict(hour=h, **m, rho_used=rho.get(h)) for h, m in meas.items()]).to_csv(
            os.path.join(d, "hourly.csv"), index=False)
        rho_new = update_rho(meas, rho, n, K, step_up)
        with open(os.path.join(base, f"rho_after_iter{n}.json"), "w") as f:
            json.dump({str(h): v for h, v in rho_new.items()}, f, indent=2)
        met = realized_metrics(network, served)
        opex_i, rate_real = operating_cost(met, K)
        with open(os.path.join(d, "stage1_info.json")) as f:
            s1info = json.load(f)
        s1h = {int(h): v for h, v in s1info.get("flights_in_capped_hours", {}).items()}
        gaps = {h: (meas[h]["served"] + meas[h]["repositioning"]) - s1h[h] for h in s1h}
        sold = float(df1["uam_pax"].sum())
        cost_gap = abs(rate - rate_real) / max(rate_real, 1e-9)
        fc = None if df_prev is None else fare_change(df1, df_prev)
        pc = None if df_prev is None else plan_change(df1, df_prev)
        n_cand = sum(m["candidates"] for m in meas.values())
        n_got = sum(m["served"] for m in meas.values())
        row = dict(iteration=n, capped_hours=len(rho), opex_per_asm_used=rate, fixed_per_flight_used=fixed,
                   opex_per_asm_realized=rate_real, cost_gap=cost_gap,
                   planned_flights=float(df1["num_flights"].fillna(0).sum()), tickets_sold=sold,
                   pax_flown=float(met["number_of_revenue_passengers"]),
                   pax_flown_over_sold=met["number_of_revenue_passengers"] / max(sold, 1e-9),
                   avg_fare=float((df1["fare"] * df1["uam_pax"]).sum() / max(sold, 1e-9)),
                   fare_change=fc, plan_change=pc, rho_change=rho_change(rho_new, rho),
                   capped_hours_checked=len(gaps), max_flight_gap=max([abs(g) for g in gaps.values()], default=0.0),
                   stage1_objective=s1info["objective"], candidates=n_cand, served_flights=n_got,
                   served_share=n_got / max(n_cand, 1), toy_daily_opex=opex_i,
                   toy_daily_profit=met["total_revenue"] - opex_i, **binfo, **met)
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(base, "convergence.csv"), index=False)
        log(f"  iter {n}: cost used ${rate:.3f} vs realized ${rate_real:.3f} (gap {cost_gap:.1%}), "
            f"fare change {'n/a' if fc is None else f'{fc:.1%}'}, plan change {'n/a' if pc is None else f'{pc:.1%}'}, "
            f"planned {row['planned_flights']:.0f}, served {n_got}/{n_cand}, pax flown/sold {row['pax_flown_over_sold']:.1%}, "
            f"max hourly flight gap {row['max_flight_gap']:.1f}, profit ${row['toy_daily_profit']:,.0f}")
        if n >= 1 and cost_gap <= tol_cost and fc <= tol_fare and pc <= tol_plan:
            log(f"  CONVERGED at iteration {n}")
            rows[-1]["converged"] = True
            pd.DataFrame(rows).to_csv(os.path.join(base, "convergence.csv"), index=False)
            break
        rho, rate, fixed, df_prev = rho_new, rate_real, 0.0, df1
    finish(base, rows)


def finish(base, rows):
    """final_iteration.txt = converged round, else the last round (Andrew, 2026-10-10). Best-profit round recorded too."""
    conv = rows[-1].get("converged", False)
    best = max(rows, key=lambda r: r["toy_daily_profit"])
    final = rows[-1]["iteration"]
    with open(os.path.join(base, "final_iteration.txt"), "w") as f:
        f.write(str(final))
    with open(os.path.join(base, "DONE.json"), "w") as f:
        json.dump(dict(converged=bool(conv), rounds=len(rows), final_iteration=int(final),
                       best_profit_iteration=int(best["iteration"]), finished=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=2)
    log(f"{'CONVERGED' if conv else 'NOT converged (reporting the last round)'}: final iteration {final}; "
        f"table -> {os.path.join(base, 'convergence.csv')}")


def market_flights(df1):
    """Stage-1 revenue flights per (origin, destination, hour); hour = (slot - 1) // 2 (slots start at 1)."""
    g = df1.assign(hour=(df1["passenger_arrival_time_slot"] - 1) // 2).groupby(
        ["origin_vertiport_id", "destination_vertiport_id", "hour"])["num_flights"].sum()
    return {(int(o), int(d), int(h)): float(v) for (o, d, h), v in g.items()}


def converge_market(K, max_iter, n_processes, tol_fare=0.02, tol_cost=0.02, pax_ok=0.95, step_up=2.0):
    """F2-7 ORIGINAL PLAN (fix2_progress, 2026-10-04/08; stop test agreed with Andrew 2026-10-09).
    Per market m = (origin, destination, hour):
      s_m = share of stage-2 candidate flights of m that stage 2 served (no candidates -> nothing unserved -> 1)
      target_m = x_m * s_m if s_m < 0.95, else x_m + 2          (x_m = stage-1 flights in m)
      cap_m: new cap = target (only when s_m < 0.95); existing cap moves by MSA, lambda = 1/(n+1)
    Cost: round 0 = $0.79/seat-mile + $20/flight (ATRDS). From round 1, stage 1's cost per flight = rate x 4 x miles
      with rate = stage 2's realized operating cost (all costs, cost.py formulas) / its seat-miles; no $20.
    Stop (n >= 1): passengers flown / tickets sold >= 0.95, |rate used - rate realized| / realized <= 2%,
      fare change <= 2% and stage-1 flights per hour change <= 2%. Else the round with the highest stage-2 profit (finish())."""
    # v2 (2026-10-09): round 0 = $0.79/seat-mile + $20/flight (ATRDS); from round 1 stage 2's cost per seat-mile, no $20
    base = R.ensure(os.path.join(OUT, run_dir_name("market", K, step_up)))
    caps, rate, fixed, rows, df_prev = {}, 0.79, FIXED_COST_PER_FLIGHT, [], None
    for n in range(max_iter):
        d = R.ensure(os.path.join(base, f"iter{n}"))
        with open(os.path.join(d, "inputs.json"), "w") as f:  # what stage 1 is given this round
            json.dump(dict(opex_per_asm=rate, fixed_cost_per_flight=fixed, caps={f"{k[0]}_{k[1]}_{k[2]}": v for k, v in sorted(caps.items())}), f, indent=1)
        log(f"===== K={K} market feedback, iteration {n}: {len(caps)} capped markets, cost ${rate:.3f}/seat-mile + ${fixed:.0f}/flight")
        s1 = os.path.join(d, "stage1.csv")
        df1 = pd.read_csv(s1) if os.path.exists(s1) else run_stage1(K, None, s1, market_caps=dict(caps) or None, opex_per_asm=rate,
                                                                 fixed_cost_per_flight=fixed)
        pkl, s2 = os.path.join(d, "network.pkl"), os.path.join(d, "stage2.csv")
        if os.path.exists(pkl) and os.path.exists(s2):
            with open(pkl, "rb") as f:
                network = pickle.load(f)
            served = pd.read_csv(s2)
            with open(os.path.join(d, "bridge_info.json")) as f:
                binfo = json.load(f)
        else:
            network, binfo = bridge(df1, PRICING_SEED, pkl, load_markets(s1))
            with open(os.path.join(d, "bridge_info.json"), "w") as f:
                json.dump(binfo, f, indent=2)
            log(f"  bridge: {binfo}")
            served = run_stage2(network, K, n_processes, s2)
        share, n_cand, n_got = served_share_by_hour(network, served)
        x = market_flights(df1)
        # ---- cap update (markets that exist in stage 1; stage-2 flights leaving after the last stage-1 hour are ignored)
        new_caps, mrows = dict(caps), []
        for key in sorted(set(x) | set(caps)):
            xm = x.get(key, 0.0)
            sm = share.get(key, 1.0)
            if xm <= 0 and key not in caps:
                continue
            target = xm * sm if sm < SHARE_OK else xm + step_up
            if key in caps:
                new_caps[key] = caps[key] + (target - caps[key]) / (n + 1)
            elif sm < SHARE_OK:
                new_caps[key] = target
            mrows.append(dict(origin=key[0], destination=key[1], hour=key[2], stage1_flights=xm, served_share=sm,
                              target=target, cap_used=caps.get(key), cap_next=new_caps.get(key)))
        pd.DataFrame(mrows).to_csv(os.path.join(d, "markets_feedback.csv"), index=False)
        # ---- cost and passengers
        met = realized_metrics(network, served)
        opex_i, rate_real = operating_cost(met, K)
        sold = float(df1["uam_pax"].sum())
        flown = float(met["number_of_revenue_passengers"])
        cost_gap = abs(rate - rate_real) / max(rate_real, 1e-9)
        fc = None if df_prev is None else fare_change(df1, df_prev)
        row = dict(iteration=n, capped_markets=len(caps), opex_per_asm_used=rate, opex_per_asm_realized=rate_real,
                   cost_gap=cost_gap, planned_flights=float(df1["num_flights"].fillna(0).sum()), tickets_sold=sold,
                   stage2_pax=binfo.get("pax"), pax_flown=flown, pax_flown_over_sold=flown / max(sold, 1e-9),
                   avg_fare=float((df1["fare"] * df1["uam_pax"]).sum() / max(sold, 1e-9)), fare_change=fc,
                   plan_change=None if df_prev is None else plan_change(df1, df_prev),
                   candidates=n_cand, served_flights=n_got, served_share=n_got / max(n_cand, 1),
                   toy_daily_opex=opex_i, toy_daily_profit=met["total_revenue"] - opex_i, **binfo, **met)
        rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(base, "convergence.csv"), index=False)
        log(f"  iter {n}: planned {row['planned_flights']:.0f} flights, sold {sold:.0f}, flown {flown:.0f} "
            f"({row['pax_flown_over_sold']:.1%}), cost used ${rate:.3f} vs realized ${rate_real:.3f} ({cost_gap:.1%}), "
            f"fare change {'n/a' if fc is None else f'{fc:.1%}'}, served {n_got}/{n_cand}, profit ${row['toy_daily_profit']:,.0f}")
        # fares settled = fares of markets in both rounds AND the set of flown markets (plan change) stable: when the seat
        # constraint binds, a market's fare depends only on its flight count, so dropped/added markets show up in plan change
        if (n >= 1 and row["pax_flown_over_sold"] >= pax_ok and cost_gap <= tol_cost and fc <= tol_fare
                and row["plan_change"] <= tol_fare):
            log(f"  CONVERGED at iteration {n}")
            rows[-1]["converged"] = True
            pd.DataFrame(rows).to_csv(os.path.join(base, "convergence.csv"), index=False)
            break
        caps, rate, fixed, df_prev = new_caps, rate_real, 0.0, df1
    finish(base, rows)


# ---------------------------------------------------------------- evaluate on other demand days
def evaluate(K, plan, seeds, n_processes, method="hourly", step_up=2.0):
    # JAT2026 (2026-10-10): same folder as the converge run of this method; no silent fallback to older runs
    base = os.path.join(OUT, run_dir_name(method, K, step_up))
    if not os.path.exists(os.path.join(base, "final_iteration.txt" if plan == "final" else plan)):
        raise FileNotFoundError(f"no converge run to evaluate in {base} (plan {plan})")
    if plan == "final":
        with open(os.path.join(base, "final_iteration.txt")) as f:
            plan_dir = f"iter{f.read().strip()}"
    else:
        plan_dir = plan
    df1 = pd.read_csv(os.path.join(base, plan_dir, "stage1.csv"))
    markets = load_markets(os.path.join(base, plan_dir, "stage1.csv"))
    rows = []
    for seed in seeds:
        d = R.ensure(os.path.join(base, f"eval_{plan_dir}", f"seed{seed}"))
        pkl, s2 = os.path.join(d, "network.pkl"), os.path.join(d, "stage2.csv")
        log(f"===== evaluate {plan_dir} on seed {seed}")
        if os.path.exists(pkl) and os.path.exists(s2):
            with open(pkl, "rb") as f:
                network = pickle.load(f)
            served = pd.read_csv(s2)
            binfo = {}
        else:
            network, binfo = bridge(df1, seed, pkl, markets)
            served = run_stage2(network, K, n_processes, s2)
        share, n_cand, n_got = served_share_by_hour(network, served)
        rows.append(dict(plan=plan_dir, seed=seed, candidates=n_cand, served_flights=n_got,
                         served_share=n_got / max(n_cand, 1), **binfo, **realized_metrics(network, served)))
        pd.DataFrame(rows).to_csv(os.path.join(base, f"evaluation_{plan_dir}.csv"), index=False)
    res = pd.DataFrame(rows)
    cols = ["served_share", "number_of_revenue_flights", "number_of_revenue_passengers", "daily_revenue",
            "daily_profit", "RASM", "CASM"]
    print(res[["seed"] + cols].to_string(index=False))
    print(res[cols].agg(["mean", "min", "max"]).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["converge", "evaluate"])
    ap.add_argument("--fleet", type=int, default=30)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--plan", default="final", help="'final' or an iteration folder such as iter0 (open loop)")
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--n-processes", type=int, default=10)
    ap.add_argument("--vrp-threads", type=int, default=2, help="Gurobi threads per stage-2 process")
    ap.add_argument("--toy", action="store_true", help="test instance: LAX, DTLA, ANH, 7-11 am, fleet 10 (~100-120 candidates)")
    ap.add_argument("--toy-hours", default="7-11")
    ap.add_argument("--toy-vertiports", default="LAX,DTLA,ANH")
    ap.add_argument("--step-up", type=float, default=2.0, help="flights a cap rises by when stage 2 served >= 95%% (both methods)")
    ap.add_argument("--cost-basis", choices=["all", "variable"], default="all",
                    help="market method: cost per seat-mile fed back = all operating costs (default) or variable costs only")
    ap.add_argument("--method", choices=["hourly", "market"], default="hourly",
                    help="hourly = Algorithm 1 revised (hourly network flight caps + cost feedback); market = Algorithm 0")
    ap.add_argument("--bridge", choices=["hourly", "thin"], default="hourly",
                    help="hourly = ATRDS share array (time-shifted for trips to LAX); thin = per-passenger, own-slot share")
    a = ap.parse_args()
    os.environ["JAT_VRP_THREADS"] = str(a.vrp_threads)
    global BRIDGE, OUT, COST_BASIS
    BRIDGE = a.bridge
    COST_BASIS = a.cost_basis
    if a.toy:
        enable_toy(hours=tuple(int(x) for x in a.toy_hours.split("-")),
                   vertiports=tuple(a.toy_vertiports.split(",")))
        if a.fleet == 30:
            a.fleet = 10
    if BRIDGE == "thin":
        OUT = OUT + "_thin"
    log(f"output folder: {OUT}")
    if a.step == "converge":
        if a.method == "market":
            converge_market(a.fleet, a.max_iter, a.n_processes, step_up=a.step_up)
        else:
            converge(a.fleet, a.max_iter, a.n_processes, step_up=a.step_up)
    else:
        lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds),) * 2
        evaluate(a.fleet, a.plan, list(range(lo, hi + 1)), a.n_processes, a.method, a.step_up)


if __name__ == "__main__":
    main()
