"""JAT2026: environment check for the ATRDS reproduction (F2-0).

Run from the repo root with the wisk_ram_study venv:
    C:\\work_python\\wisk\\wisk_ram_study\\.venv\\Scripts\\python.exe scripts\\jat2026\\check_env.py
"""
import importlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

print("python", sys.version.split()[0], "|", sys.executable)
for mod in ["numpy", "pandas", "scipy", "geopy", "tqdm", "matplotlib", "seaborn", "numpy_financial", "gurobipy"]:
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:16s} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {mod:16s} {e}")

# Gurobi license: a 3,000-variable model exceeds the pip restricted license (2,000)
import gurobipy as gp
try:
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVars(3000)
    m.setObjective(x.sum())
    m.optimize()
    print("  ok   Gurobi full license", gp.gurobi.version())
except gp.GurobiError as e:
    print("  FAIL Gurobi license:", e)

# Data files
data = os.path.join(ROOT, "uam_system_model", "data")
for f in ["LAX_ind.csv", "T_F41SCHEDULE_B43.csv", "2-travel_time.pkl",
          "config/flight_time_matrix.npy", "config/energy_consumption_matrix.npy",
          "config/financial_analysis_params.json", "config/p_uam_array_10min_wait_deterministic.npy"]:
    print(("  ok   " if os.path.exists(os.path.join(data, f)) else "  MISS ") + f)

# Package import
for mod in ["uam_system_model.StarNetwork", "uam_system_model.Pricing",
            "uam_system_model.PricingStatic", "uam_system_model.FleetOpVRP"]:
    try:
        importlib.import_module(mod)
        print("  ok   import", mod)
    except Exception as e:  # noqa: BLE001
        print("  FAIL import", mod, "->", repr(e))
