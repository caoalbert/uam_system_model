"""JAT2026: evidence of the stage-1 / stage-2 gap from a converge run (toy or full).
Usage: python scripts/jat2026/toy_evidence.py runs/jat_toy_7-17/variable_K10 [--fleet 10]
Writes evidence_hourly.csv and evidence_aircraft.txt in the run folder; prints a summary.

Per hour and iteration:
  sold      = UAM tickets sold in stage 1 (sum theta_i d_i)
  plan_fl   = flights planned by stage 1
  cand_pax / cand_fl = passengers / flights handed to stage 2 (new random day, dispatch rule)
  flown_pax / flown_fl = passengers / revenue flights stage 2 actually flies
  fly / charge / idle = share of fleet time in tours / in gaps with charging / in gaps without charging
  soc_dep   = mean battery level (%) at tour departure (stage-2 SoC floor is 20%)
"""
import os, sys, pickle, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def split_minutes(t0, t1, acc):
    while t0 < t1:
        h = int(min(t0 // 60, 23)); nxt = min(t1, (int(t0 // 60) + 1) * 60)
        acc[h] = acc.get(h, 0.0) + (nxt - t0); t0 = nxt


def one_iter(d, K):
    s1 = pd.read_csv(os.path.join(d, "stage1.csv"))
    s1["hour"] = (s1["passenger_arrival_time_slot"] - 1) // 2
    net = pickle.load(open(os.path.join(d, "network.pkl"), "rb"))
    sc = net.schedule.copy(); sc["hour"] = (sc["schedule"] // 60).astype(int)
    s2 = pd.read_csv(os.path.join(d, "stage2.csv")) if os.path.exists(os.path.join(d, "stage2.csv")) else pd.DataFrame()
    vd = net.vertiport_dict; ft = np.asarray(net.flight_time) * 5
    flown_pax, flown_fl, fly, chg, idle, soc = {}, {}, {}, {}, {}, {}
    lines = []
    if len(s2):
        for t, seq, npax in zip(s2["time"], s2["tour_sequence"], s2["num_pax"]):
            legs = [vd[x] for x in str(seq).split("-")]
            for a, b in zip(legs[:-1], legs[1:]):
                h = int(t // 60); flown_fl[h] = flown_fl.get(h, 0) + 1; t += ft[a, b]
        for t, n in zip(s2["time"], s2["num_pax"]):
            flown_pax[int(t // 60)] = flown_pax.get(int(t // 60), 0) + n
        # stage-2 battery variables (FleetOpVRP Eqs. 5-42): p = SoC on landing after the tour, q = SoC after
        # charging at the tour's end, p2/q2 = same after the repositioning flight. Departure SoC = p + tour energy.
        s2 = s2.assign(dep=s2["p"] + s2["energy_consumption"], gain=(s2["q"] - s2["p"]) + (s2["q2"] - s2["p2"]))
        for t, q in zip(s2["time"], s2["dep"]):
            soc.setdefault(int(t // 60), []).append(q)
        for ac, g in s2.sort_values("time").groupby("cluster_id"):
            prev = None; row = [f"aircraft {ac}:"]
            for t, dur, dep, land, gain, seq in zip(g["time"], g["duration"], g["dep"], g["p"], g["gain"], g["tour_sequence"]):
                if prev is not None:
                    gap0, pg = prev
                    if t > gap0:
                        split_minutes(gap0, t, chg if pg > 0.5 else idle)
                        row.append(f"[{'charge +%.0f%%' % pg if pg > 0.5 else 'no charge'} {t - gap0:.0f}m]")
                split_minutes(t, t + dur, fly)
                row.append(f"{int(t//60):02d}:{int(t%60):02d} {seq} ({dep:.0f}->{land:.0f}%)")
                prev = (t + dur, gain)
            lines.append(" ".join(row))
    hours = sorted(set(s1["hour"]) | set(sc["hour"]))
    rows = []
    for h in hours:
        rows.append(dict(hour=h, sold=s1.loc[s1.hour == h, "uam_pax"].sum(), plan_fl=s1.loc[s1.hour == h, "num_flights"].sum(),
                         cand_pax=sc.loc[sc.hour == h, "num_pax"].sum(), cand_fl=int((sc.hour == h).sum()),
                         flown_pax=flown_pax.get(h, 0), flown_fl=flown_fl.get(h, 0),
                         fly=fly.get(h, 0) / (K * 60), charge=chg.get(h, 0) / (K * 60), idle=idle.get(h, 0) / (K * 60),
                         soc_dep=np.mean(soc[h]) if h in soc else np.nan))
    return pd.DataFrame(rows), lines


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--fleet", type=int, default=10)
    a = ap.parse_args()
    its = sorted([x for x in os.listdir(a.run_dir) if x.startswith("iter")], key=lambda x: int(x[4:]))
    allr, txt = [], []
    pd.set_option("display.width", 200)
    for it in its:
        d = os.path.join(a.run_dir, it)
        if not os.path.exists(os.path.join(d, "network.pkl")):
            continue
        df, lines = one_iter(d, a.fleet); df.insert(0, "iter", int(it[4:])); allr.append(df)
        txt += [f"===== {it} ====="] + lines
        tot = df[["sold", "plan_fl", "cand_pax", "cand_fl", "flown_pax", "flown_fl"]].sum()
        print(f"\n===== {it}: sold {tot.sold:.0f} | planned {tot.plan_fl:.0f} fl | candidates {tot.cand_pax:.0f} pax / {tot.cand_fl:.0f} fl"
              f" | flown {tot.flown_pax:.0f} pax / {tot.flown_fl:.0f} fl ({tot.flown_pax / max(tot.cand_pax, 1):.0%} of candidate pax)")
        print(df.drop(columns="iter").round(2).to_string(index=False))
    pd.concat(allr).to_csv(os.path.join(a.run_dir, "evidence_hourly.csv"), index=False)
    open(os.path.join(a.run_dir, "evidence_aircraft.txt"), "w").write("\n".join(txt))
    print(f"\nwrote {a.run_dir}/evidence_hourly.csv and evidence_aircraft.txt")


if __name__ == "__main__":
    main()
