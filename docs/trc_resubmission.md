# TRC-25-02500 Rejection and Resubmission Plan

**Paper:** *The Urban Air Mobility Fleet Scheduling Problem with Nonlinear Charging Time*
**Manuscript:** TRC-25-02500
**Decision:** Rejected by *Transportation Research Part C* on 2026-02-12 (Editor: Lishuai Li) citing "lack of sufficient novelty"
**Authors:** S. Cao, E. B. Onat, X. Jiang, R. Sengupta, C. Yan, M. Hansen

This document maps each reviewer critique to the specific code changes on the `refactor/vrp-algorithmic-improvements` branch, and lists the follow-up experiments and paper rewrites still needed before a resubmission attempt.

---

## Reviewer 1 — Critiques and Status

### R1.1 — No small-scale MIP benchmark
> *"The authors do not solve small-scale instances of the full eUAMVRP-NL model to optimality... it is difficult to assess the computational properties of the proposed formulation."*

**Code change:** `_RunVRP` in `uam_system_model/FleetOpVRP.py` has been refactored to accept `K` (number of vehicles), `time_limit`, and `optimality_gap` as parameters rather than hardcoding `K = 1`. A new method

```python
FleetOpVRP.solve_exact(num_vehicles, time_limit=3600, optimality_gap=0.01)
```

runs the full multi-vehicle eUAMVRP-NL MIP. Returns objective value, final MIP gap, runtime, and the served/unserved DataFrames. Must be called after `optimize()` has populated `self.demand2`.

**Still needed for resubmission:**
- Run `solve_exact()` on small instances (2–3 vertiports, 10–20 flight tasks, 2–3 vehicles)
- Report optimal objective, MIP gap, and runtime for each
- Compute optimality gap of CFRS against these benchmarks and include in Section 5

### R1.2 — CFRS heuristic too weak vs. state-of-the-art
> *"The literature contains many powerful metaheuristics such as ILS/AILS, ALNS, and various matheuristics... In contrast, the CFRS framework does not incorporate advanced neighborhood search, adaptive mechanisms, or intensification/diversification strategies."*

**Code change:** The single-pass `_crossover` post-processing has been replaced with a full Adaptive Large Neighborhood Search (`_alns` method):

| Component | Implementation |
|---|---|
| Destroy operators | `_random_removal`, `_related_removal` (temporally clustered tours) |
| Repair operators | `_greedy_insert`, `_regret2_insert` |
| Acceptance criterion | Simulated annealing with geometric cooling |
| Operator selection | Roulette wheel over adaptive weights |
| Weight update | `w ← 0.8·w + 0.2·score` each iteration |

Controlled by the `alns_iterations` parameter to `optimize()` (default 30). Set `alns_iterations=0` to revert to the legacy single-pass insertion for ablation.

**Still needed for resubmission:**
- Empirical ablation: ALNS vs. single-pass `_crossover`, report served-ratio gains and wall-clock cost
- Optionally benchmark against a reference ALNS or genetic algorithm implementation from VRP literature

---

## Reviewer 2 — Critiques and Status

### R2.1 — Piecewise-linear charging contribution unclear
Paper rewrite required. The nonlinear charging model's novelty needs to be clarified in Sections 1 and 3 — specifically, what it enables that a simpler constant-rate or single-breakpoint model cannot. No code change needed.

### R2.2 — CFRS validated only on two-vertiport network; IPFS benchmark inappropriate
> *"It is unclear how this IPFS can provide an optimal benchmark for the current VRP formulation."*

**Still needed for resubmission:**
- Drop the IPFS comparison in Section 5, replace with `solve_exact()` output on matching small instances
- Run a scalability study on the full LAX network (5 vertiports) at fleet sizes of 5, 10, 15 vehicles
- Optionally extend to JFK and Beijing (both supported by the existing `StarNetworkJFK` / `StarNetworkBJ` classes)

### R2.3 — Formulation concerns (Section 3)

| # | Reviewer concern | Code change |
|---|---|---|
| 3.1 | Second charging session when repositioning time is zero | Added Big-M constraint: when `tilt[i, j] == 0`, `v_ik2 - w_ik2 ≤ M·(1 - x_ijk)`, forcing the second charging session duration to zero whenever the edge is used and repo time is zero |
| 3.2 | `Delta_i^k` and `delta_i^k` are redundant (determined by SoC increment) | Removed both as Gurobi decision variables; inlined `(v_ik - w_ik)` and `(v_ik2 - w_ik2)` directly into constraint (36). Also removed their defining constraints (33, 34) and the orphaned `Delta_ik >= 0` domain constraint. Added previously missing domain bounds `w_ik2, v_ik2 >= 0` |
| 3.3 | Intermediate SoC level consistency not enforced | Added contiguity ordering constraints inside the existing per-`s` loop: `m[s] >= m[s-1] - m[s+1]` for all four breakpoint families (`m_isk`, `n_isk`, `m_isk2`, `n_isk2`). Prevents non-contiguous SoC level selections |
| 3.4 | Section 3.4 doesn't delineate novel VRP adaptations | Paper rewrite: flag each constraint as *classical VRP*, *charging extension*, or *UAM-specific* |

### R2.4 — Heuristic concerns (Section 4)

| # | Concern | Status |
|---|---|---|
| 4.1 | Single-vehicle decomposition reduces to TSP, misses multi-vehicle interactions | Partially addressed: ALNS reintroduces global coordination across vehicles via destroy/repair across clusters. Multi-vehicle interaction effects can also be measured by comparing `solve_exact(K>1)` to the CFRS result |
| 4.2 | No comparison with established metaheuristics (ALNS, GA) | Partially addressed in code (ALNS is now the default); further benchmarking against external reference implementations optional |
| 4.3 | Section 4.1 "DFS" subtour-building unclear; "sub-tour" terminology overloaded | Paper rewrite: the procedure in `_BuildSubtours` is actually a **greedy forward chain-builder**, not DFS. Also rename "sub-tour" to "flight chain" or similar to avoid collision with the classical VRP meaning |
| 4.4 | Insertion procedure ("adding constraints") insufficient | Paper rewrite: explain the `resolve=True` path — previously served tours become constrained as `x_ijk.sum(i,*,*) == 1`, and the solver is re-run with unserved tours admitted as candidates |

### R2.5 — Notation inconsistencies
Paper rewrite. Specifics to fix:
- `S` used for both the SoC-level set and the tuple element — rename tuple element
- `t` used for both time and an index over `Ĩ` — use a distinct index letter
- `x` used for both VRP decision variable `x_{ij}^k` and knapsack variable `x_s` — rename knapsack variable
- `n` reused across formulations — disambiguate

---

## Checklist before resubmission

- [ ] Run `solve_exact()` on ≥5 small instances; report obj, gap, runtime (R1.1)
- [ ] Run ALNS vs. single-pass ablation; report Δ served-ratio and Δ runtime (R1.2)
- [ ] Run 5-vertiport LAX scalability study (R2.2)
- [ ] Replace IPFS comparison in Section 5 with `solve_exact()` benchmark (R2.2)
- [ ] Rewrite Sections 1 and 3 to clarify nonlinear-charging contribution (R2.1)
- [ ] Rewrite Section 3.4 distinguishing VRP/charging/UAM-specific constraints (R2.3.4)
- [ ] Rewrite Section 4.1 describing the actual subtour-building algorithm; rename "sub-tour" (R2.4.3)
- [ ] Rewrite Section 4 "adding constraints" / insertion procedure (R2.4.4)
- [ ] Fix notation collisions (R2.5)

## File-level pointers

| Change | File | Method / line |
|---|---|---|
| `solve_exact`, `_alns`, destroy/repair ops | [`uam_system_model/FleetOpVRP.py`](../uam_system_model/FleetOpVRP.py) | `FleetOpVRP.solve_exact`, `FleetOpVRP._alns`, `_random_removal`, `_related_removal`, `_greedy_insert`, `_regret2_insert` |
| Formulation fixes (Delta removal, zero-repo, SoC contiguity) | [`uam_system_model/FleetOpVRP.py`](../uam_system_model/FleetOpVRP.py) | `FleetOpVRP._RunVRP` |
