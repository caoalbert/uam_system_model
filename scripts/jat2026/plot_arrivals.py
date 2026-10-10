"""JAT2026: passenger arrivals at each vertiport, stage 1 vs stage-2 input, per 30-min slot.

For each vertiport v, passengers who START their UAM trip at v (LAX: trips from LAX; spokes: trips to LAX):
  * stage 1 (tickets sold)      = sum over markets (slot, o = v) of theta_i * d_i   (from stage1.csv)
  * stage 2 input, old bridge   = UAM passengers stage 2 actually received (network.pkl of that run)
  * stage 2 input, thin bridge  = rebuilt here from the same stage1.csv with run_jat.bridge(..., BRIDGE="thin")
Stage 2 uses the pricing seed in converge runs, so the thin-bridge passengers are a random subset of the
exact market day stage 1 priced; only the per-passenger UAM choice is random.

Usage (repo root):
  python scripts/jat2026/plot_arrivals.py runs/jat/variable_K30/iter0
  python scripts/jat2026/plot_arrivals.py runs/jat_toy/variable_K10/iter0 --toy-hours 7-11
Output: <iter dir>/arrival_plots/arrivals_<VERTIPORT>.png, arrivals_all.png, arrivals_by_slot.csv
"""
import argparse, os, pickle, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_jat as J  # noqa: E402
R = J.R


def by_slot_from_pax(pa):
    slot = np.ceil(pa["passenger_arrival_time_s"] / 1800).clip(1, 48).astype(int)
    return pa.assign(slot=slot).groupby(["origin_vertiport_id", "slot"]).size()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("iter_dir")
    ap.add_argument("--toy-hours", default=None, help="e.g. 7-11 if the run was a toy run")
    ap.add_argument("--toy-vertiports", default="LAX,DTLA,ANH")
    ap.add_argument("--seed", type=int, default=J.PRICING_SEED, help="stage-2 demand seed used by that run")
    a = ap.parse_args()
    if a.toy_hours:
        J.enable_toy(hours=tuple(int(x) for x in a.toy_hours.split("-")), vertiports=tuple(a.toy_vertiports.split(",")))
    out = R.ensure(os.path.join(a.iter_dir, "arrival_plots"))
    s1 = pd.read_csv(os.path.join(a.iter_dir, "stage1.csv"))
    sold = s1.groupby(["origin_vertiport_id", "passenger_arrival_time_slot"])["uam_pax"].sum()
    series = {"stage 1: tickets sold": sold}
    pkl = os.path.join(a.iter_dir, "network.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            net_old = pickle.load(f)
        name = "stage 2 input (run's bridge)"
        series[name] = by_slot_from_pax(net_old.pax_arrival_times)
    J.BRIDGE = "thin"
    net_thin, _ = J.bridge(s1, a.seed, os.path.join(out, "network_thin.pkl"))  # saved for inspection
    series["stage 2 input (thin bridge)"] = by_slot_from_pax(net_thin.pax_arrival_times)

    names = R.VERTIPORTS
    vs = sorted(set().union(*[set(s.index.get_level_values(0)) for s in series.values()]))
    slots = np.arange(1, 49); x = (slots - 0.5) / 2  # slot centre in hours
    rows = []
    styles = {"stage 1: tickets sold": dict(color="k", lw=2), "stage 2 input (run's bridge)": dict(color="tab:red", ls="--"),
              "stage 2 input (thin bridge)": dict(color="tab:blue")}
    fig_all, axs = plt.subplots(int(np.ceil(len(vs) / 3)), 3, figsize=(15, 3.2 * int(np.ceil(len(vs) / 3))), squeeze=False)
    for k, v in enumerate(vs):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for lab, s in series.items():
            y = np.array([float(s.get((v, sl), 0.0)) for sl in slots])
            for sl, yy in zip(slots, y):
                rows.append(dict(vertiport=names[v], slot=sl, series=lab, pax=yy))
            for a_ in (ax, axs.flat[k]):
                a_.step(x, y, where="mid", label=f"{lab} (day {y.sum():.0f})", **styles[lab])
        trip = "trips from LAX" if v == 0 else "trips to LAX"
        for a_ in (ax, axs.flat[k]):
            a_.set_title(f"{names[v]} ({trip})"); a_.set_xlim(0, 24); a_.set_xticks(range(0, 25, 3)); a_.grid(alpha=0.3)
        ax.set_xlabel("passenger arrival time at vertiport (h)"); ax.set_ylabel("UAM passengers per 30 min")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, f"arrivals_{names[v]}.png"), dpi=130); plt.close(fig)
        axs.flat[k].legend(fontsize=7)
    for a_ in list(axs.flat)[len(vs):]:
        a_.axis("off")
    fig_all.supxlabel("passenger arrival time at vertiport (h)"); fig_all.supylabel("UAM passengers per 30 min")
    fig_all.tight_layout(); fig_all.savefig(os.path.join(out, "arrivals_all.png"), dpi=110); plt.close(fig_all)
    tab = pd.DataFrame(rows).pivot_table(index=["vertiport", "slot"], columns="series", values="pax").reset_index()
    tab.to_csv(os.path.join(out, "arrivals_by_slot.csv"), index=False)
    tot = tab.groupby("vertiport")[list(series)].sum().round(0)
    print(tot.to_string()); print(f"plots -> {out}")


if __name__ == "__main__":
    main()
