"""
Dobda Case — Task 8
Author: ChatGPT

Goal:
  Compare two factory production strategies:
    - Strategy A (Chase): produce to meet daily target
    - Strategy B (Steady): produce at one constant daily rate

Inputs:
  - FC Daily {label}.csv (from Task 7)
  - df_global2_seasonality_daily.csv (daily demand from Task 2)

Outputs:
  - CSV files: Strategy A, Strategy B, Summary
  - Excel: Task8 Report + Task8 Combined
  - PNG charts: AF production (daily/cumulative), DC inventory (daily/cumulative)

Steps:
  1. Read Task 7 daily targets and Task 2 demand
  2. Merge them into one daily table
  3. Run Strategy A (produce to target) and Strategy B (steady rate)
  4. Save daily results, summary, and charts
"""

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Where outputs go
OUT_DIR = Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
INIT_INV  = 0.0        # starting DC inventory (overridden later by first target)
AGG_NET   = False      # True = sum all FCs, False = use one FC
FC_FILTER: Optional[str] = None  # choose one FC if not aggregating

# Task 2 demand source
TASK2_CSV      = Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task1_2/df_global2_seasonality_daily.csv")
TASK2_VAL_CAND: List[str] = [
    "units_2026_expected_distnormal_99",
    "units_2026_expected_robust99",
    "demand_2026_robust99",
    "daily_demand_99",
]

# -------- Helpers --------
def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """Pick the first column that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"Missing: {candidates}")
    return None

def to_num(s: pd.Series) -> pd.Series:
    """Make numeric, replace bad values with 0, no negatives."""
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)

def _steady_rate(d: np.ndarray, t: np.ndarray, init_inv: float) -> float:
    """Find smallest constant rate so DC never drops below target."""
    if len(d) == 0:
        return 0.0
    cum_d_prev = np.concatenate(([0.0], np.cumsum(d)[:-1]))
    rate_cands = (t - init_inv + cum_d_prev) / (np.arange(len(d)) + 1)
    rate_cands = np.nan_to_num(rate_cands, nan=0.0, posinf=0.0, neginf=0.0)
    return float(max(0.0, rate_cands.max()))

def _safe_sheet_name(name: str) -> str:
    """Fix names so Excel accepts them."""
    return re.sub(r'[:\\/?*\[\]]', "-", name)[:31]

def _bundle_label_csvs_to_excel(label: str) -> Path:
    """Combine all CSVs for this label into one Excel file."""
    csvs = sorted(OUT_DIR.glob(f"*{label}*.csv"))
    out_xlsx = OUT_DIR / f"Task8 Combined {label}.xlsx"
    if not csvs:
        return out_xlsx
    with pd.ExcelWriter(out_xlsx) as xw:
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path)
                df.to_excel(xw, sheet_name=_safe_sheet_name(csv_path.stem), index=False)
            except Exception as e:
                pd.DataFrame({"error": [str(e)]}).to_excel(xw, sheet_name=_safe_sheet_name(f"ERR_{csv_path.stem}"), index=False)
    return out_xlsx

# -------- Core --------
def run_task8(IN_CSV: Path, label: str) -> None:
    """Run Task 8 for one config (e.g., 15FC)."""
    # Load Task 7 file
    df = pd.read_csv(IN_CSV)
    df.columns = [str(c) for c in df.columns]

    col_date   = pick_col(df, ["Date", "date"])
    col_fc     = pick_col(df, ["FC ID", "FC_ID", "fc_id"], required=False)
    col_target = pick_col(df, ["FC Target", "fc_target"])
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    df = df.dropna(subset=[col_date])

    # Aggregate FCs if needed
    if AGG_NET:
        daily = df.groupby(col_date, as_index=False)[[col_target]].sum().sort_values(col_date)
    else:
        if FC_FILTER and (col_fc in df.columns):
            sub = df[df[col_fc] == FC_FILTER].copy()
            daily = sub if not sub.empty else df.copy()
        else:
            daily = df.copy()
        daily = daily.sort_values(col_date)

    daily = daily.rename(columns={col_date: "Date", col_target: "Target"})[["Date", "Target"]]
    daily["Target"] = to_num(daily["Target"]).reset_index(drop=True)

    if daily.empty:
        raise RuntimeError(f"[{label}] No daily rows.")

    # Load Task 2 demand
    df2 = pd.read_csv(TASK2_CSV)
    col2_date = pick_col(df2, ["Date", "date", "day"])
    df2[col2_date] = pd.to_datetime(df2[col2_date], errors="coerce")
    df2 = df2.dropna(subset=[col2_date])
    col2_val = pick_col(df2, TASK2_VAL_CAND)

    t2 = df2[[col2_date, col2_val]].rename(columns={col2_date: "Date", col2_val: "Demand"}).sort_values("Date")
    daily = daily.merge(t2, on="Date", how="left")
    daily["Demand"] = to_num(daily["Demand"])

    # Arrays
    d = daily["Demand"].to_numpy()
    t = daily["Target"].to_numpy()
    n = len(d)

    # Initial inv = day 1 target
    INIT_INV_EFF = float(t[0]) if n else 0.0

    # --- Strategy A (Chase) ---
    inv_beg = np.zeros(n + 1); inv_beg[0] = INIT_INV_EFF
    af_prod = np.zeros(n); dc_mid = np.zeros(n)
    for i in range(n):
        produce = max(0.0, t[i] - inv_beg[i])
        af_prod[i] = produce
        dc_mid[i]  = inv_beg[i] + produce
        inv_beg[i+1] = max(0.0, dc_mid[i] - d[i])
    peak_cap_a, peak_mid_a, peak_eod_a = af_prod.max(), dc_mid.max(), inv_beg[1:].max()
    cap_res = np.full(n, peak_cap_a)

    # --- Strategy B (Steady) ---
    rate = _steady_rate(d, t, INIT_INV_EFF)
    inv_beg_b = np.zeros(n + 1); inv_beg_b[0] = INIT_INV_EFF
    dc_mid_b = np.zeros(n)
    for i in range(n):
        dc_mid_b[i] = inv_beg_b[i] + rate
        inv_beg_b[i+1] = max(0.0, dc_mid_b[i] - d[i])
    peak_cap_b, peak_mid_b, peak_eod_b = rate, dc_mid_b.max(), inv_beg_b[1:].max()

    # Save results
    df_a = pd.DataFrame({"Date": daily["Date"], "Demand": d, "Target": t,
                         "AF Production": af_prod, "Strategy A Capacity Reserved": cap_res,
                         "DC Inventory": dc_mid, "DC Inventory EOD": inv_beg[1:]})
    df_b = pd.DataFrame({"Date": daily["Date"], "Demand": d, "Target": t,
                         "AF Steady Rate": rate, "DC Inventory": dc_mid_b, "DC Inventory EOD": inv_beg_b[1:]})
    df_sum = pd.DataFrame({"Peak Loads": ["AF Peak Daily", "DC Peak", "DC Peak EOD"],
                           "Follow Demand": [peak_cap_a, peak_mid_a, peak_eod_a],
                           "Steady Rate":   [peak_cap_b, peak_mid_b, peak_eod_b]})

    p_a, p_b, p_s, p_x = (OUT_DIR / f"Task8 Strategy A {label}.csv",
                          OUT_DIR / f"Task8 Startegy B {label}.csv",
                          OUT_DIR / f"Task8 AF Startegy Summary {label}.csv",
                          OUT_DIR / f"Task8 Report {label}.xlsx")
    df_a.to_csv(p_a, index=False); df_b.to_csv(p_b, index=False); df_sum.to_csv(p_s, index=False)
    with pd.ExcelWriter(p_x) as xw:
        df_a.to_excel(xw, sheet_name="Strategy A", index=False)
        df_b.to_excel(xw, sheet_name="Strategy B", index=False)
        df_sum.to_excel(xw, sheet_name="Summary", index=False)

    # Plots
    plt.figure(); plt.plot(df_a["Date"], df_a["AF Production"], label="Chase")
    plt.plot(df_b["Date"], df_b["AF Steady Rate"], label="Steady")
    plt.plot(df_a["Date"], df_a["Strategy A Capacity Reserved"], "--", label="Cap Reserved")
    plt.legend(); plt.title(f"Daily AF — {label}"); plt.savefig(OUT_DIR / f"Daily AF {label}.png"); plt.close()

    plt.figure(); plt.plot(df_a["Date"], df_a["AF Production"].cumsum(), label="Chase cum")
    plt.plot(df_b["Date"], df_b["AF Steady Rate"].cumsum(), label="Steady cum")
    plt.plot(df_a["Date"], np.arange(1, n+1) * cap_res[0], "--", label="Cap Reserved cum")
    plt.legend(); plt.title(f"Cumulative AF — {label}"); plt.savefig(OUT_DIR / f"Cumulative AF {label}.png"); plt.close()

    plt.figure(); plt.plot(df_a["Date"], df_a["DC Inventory"], label="Chase")
    plt.plot(df_b["Date"], df_b["DC Inventory"], label="Steady")
    plt.legend(); plt.title(f"Daily DC Inv — {label}"); plt.savefig(OUT_DIR / f"Daily DC Inventory {label}.png"); plt.close()

    plt.figure(); plt.plot(df_a["Date"], df_a["DC Inventory"].cumsum(), label="Chase cum")
    plt.plot(df_b["Date"], df_b["DC Inventory"].cumsum(), label="Steady cum")
    plt.legend(); plt.title(f"Cumulative DC Inv — {label}"); plt.savefig(OUT_DIR / f"Cumulative DC Inventory {label}.png"); plt.close()

    # Bundle all CSVs
    bundle_path = _bundle_label_csvs_to_excel(label)
    print(f"[Task8] {label} done. CSV/Excel/PNG saved in {OUT_DIR}")

# Run for configs
CONFIGS = {
    "15FC": Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7/FC Daily 15FC.csv"),
    "4FC":  Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7/FC Daily 4FC.csv"),
    "1FC":  Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7/FC Daily 1FC.csv"),
}

if __name__ == "__main__":
    try:
        for label, path in CONFIGS.items():
            AGG_NET = label in ["4FC", "15FC"]
            run_task8(path, label)
        print("[Task8] All configs done.")
    except Exception as e:
        print(f"[Task8] ERROR: {e}", file=sys.stderr)
        raise
