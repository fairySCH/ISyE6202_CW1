"""
Task 8 — AF Planning vs Inventory Behavior (Follow-Demand vs Steady-Rate)

Refactor goals (no behavior change):
- Keep ALL outputs identical: CSV/Excel filenames, PNG filenames, plot titles.
- Keep the same column choices and math.
- Improve readability with docstrings, comments, and small guardrails.
"""

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # for PNG plots


# ========= OUTPUT PATH (unchanged) =========
OUT_DIR = Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= PARAMS (unchanged semantics) =========
INIT_INV  = 0.0        # base param (overridden by first day target when computing INIT_INV_EFF)
AGG_NET   = False      # True: aggregate across all FCs (network); False: one FC
FC_FILTER: Optional[str] = None  # e.g., "FC_01" (only if AGG_NET is False)

# ========= Task-2 demand source (unchanged paths) =========
TASK2_CSV      = Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task1_2/df_global2_seasonality_daily.csv")
TASK2_VAL_CAND: List[str] = [
    "units_2026_expected_distnormal_99",
    "units_2026_expected_robust99",
    "demand_2026_robust99",
    "daily_demand_99",
]


# ========= HELPERS =========
def pick_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    """
    Return the first existing column from `candidates` found in `df`.
    If required and none found, raise a clear error.
    """
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"Missing any of columns: {candidates}")
    return None


def to_num(s: pd.Series) -> pd.Series:
    """
    Convert to numeric, coerce NaNs to 0, clamp negatives up to 0.
    The model assumes non-negative demand/targets/inventory.
    """
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)


def _steady_rate(d: np.ndarray, t: np.ndarray, init_inv: float) -> float:
    """
    Smallest constant AF rate that never lets DC EOD inventory go negative
    while trying to respect the daily target t.

    Derivation:
        I_i = init_inv + i*r - sum_{k<i} d_k
        M_i = I_i + r >= t_i
        => r >= (t_i - init_inv + sum_{k<i} d_k) / (i+1)
    We take the max across i to ensure feasibility for the whole horizon.
    """
    if len(d) == 0:
        return 0.0
    cum_d_prev = np.concatenate(([0.0], np.cumsum(d)[:-1]))  # sum_{k<i} d_k
    rate_cands = (t - init_inv + cum_d_prev) / (np.arange(len(d)) + 1)
    rate_cands = np.nan_to_num(rate_cands, nan=0.0, posinf=0.0, neginf=0.0)
    return float(max(0.0, rate_cands.max()))


# ========= CORE LOGIC =========
def run_task8(IN_CSV: Path, label: str) -> None:
    """
    Simulate two AF strategies against Task 7 targets:
      A) Follow-Demand (Chase): produce to reach daily target before demand.
      B) Steady-Rate (Level): constant rate chosen by running-max feasibility.

    Exports:
      - Task8 Strategy A {label}.csv
      - Task8 Startegy B {label}.csv
      - Task8 AF Startegy Summary {label}.csv
      - Task8 Report {label}.xlsx
      - Daily AF {label}.png
      - Cumulative AF {label}.png
      - Daily DC Inventory {label}.png
      - Cumulative DC Inventory {label}.png
    """
    # --- Load Task 7 per-FC targets file ---
    df = pd.read_csv(IN_CSV)
    # Normalize column names for robust matching
    df.columns = [c if isinstance(c, str) else str(c) for c in df.columns]

    col_date   = pick_col(df, ["Date", "date"])
    col_fc     = pick_col(df, ["FC ID", "FC_ID", "fc_id"], required=False)
    col_target = pick_col(df, ["FC Target", "fc_target"])
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")

    # Drop any rows with invalid dates to avoid alignment surprises
    df = df.dropna(subset=[col_date])

    # --- Scope: network aggregation vs single FC (unchanged semantics) ---
    if AGG_NET:
        # Aggregate across all FCs by date (sum the target column only, as in original)
        grouped = (
            df.groupby(col_date, as_index=False)[[col_target]]
              .sum()
              .sort_values(col_date)
        )
        daily = grouped
    else:
        # If filtering to one FC is requested and the column exists
        if FC_FILTER and (col_fc in df.columns):
            sub = df[df[col_fc] == FC_FILTER].copy()
            daily = sub if not sub.empty else df.copy()
        else:
            daily = df.copy()
        daily = daily.sort_values(col_date)

    # Standardize working columns: Date / Target
    daily = daily.rename(columns={col_date: "Date", col_target: "Target"})[["Date", "Target"]]
    daily["Target"] = to_num(daily["Target"])
    daily = daily.sort_values("Date").reset_index(drop=True)

    if daily.empty:
        raise RuntimeError(f"[{label}] No daily rows to process after preprocessing.")

    # --- Load Task-2 daily demand (unchanged source) ---
    df2 = pd.read_csv(TASK2_CSV)
    col2_date = pick_col(df2, ["Date", "date", "day", "Day"])
    df2[col2_date] = pd.to_datetime(df2[col2_date], errors="coerce")
    df2 = df2.dropna(subset=[col2_date])
    col2_val = pick_col(df2, TASK2_VAL_CAND)

    t2 = (
        df2[[col2_date, col2_val]]
        .rename(columns={col2_date: "Date", col2_val: "Demand"})
        .sort_values("Date")
    )
    daily = daily.merge(t2, on="Date", how="left")
    daily["Demand"] = to_num(daily["Demand"])

    # --- NumPy views for the simulation ---
    d = daily["Demand"].to_numpy(dtype=float)
    t = daily["Target"].to_numpy(dtype=float)
    n = len(d)

    # ========= Start with Day-1 target as initial inventory (unchanged) =========
    INIT_INV_EFF = float(t[0]) if n else 0.0

    # ========= Strategy A: Follow Demand (Chase) =========
    inv_beg = np.zeros(n + 1, dtype=float)
    inv_beg[0] = INIT_INV_EFF

    af_prod_chase      = np.zeros(n, dtype=float)
    dc_inv_mid_chase   = np.zeros(n, dtype=float)

    for i in range(n):
        # Produce just enough to reach the target before demand hits
        produce_today       = max(0.0, t[i] - inv_beg[i])
        af_prod_chase[i]    = produce_today
        dc_inv_mid_chase[i] = inv_beg[i] + produce_today
        # End-of-day after serving demand (no backorders)
        inv_beg[i + 1]      = max(0.0, dc_inv_mid_chase[i] - d[i])

    peak_cap_a = float(af_prod_chase.max()) if n else 0.0
    peak_mid_a = float(dc_inv_mid_chase.max()) if n else 0.0
    peak_eod_a = float(inv_beg[1:].max()) if n else 0.0
    # Flat capacity reference line at peak daily AF
    strategy_a_capacity_reserved = np.full(n, peak_cap_a, dtype=float)

    # ========= Strategy B: Steady =========
    rate = _steady_rate(d, t, INIT_INV_EFF)

    inv_beg_b = np.zeros(n + 1, dtype=float)
    inv_beg_b[0] = INIT_INV_EFF
    dc_inv_mid_level = np.zeros(n, dtype=float)

    for i in range(n):
        dc_inv_mid_level[i] = inv_beg_b[i] + rate
        inv_beg_b[i + 1]    = max(0.0, dc_inv_mid_level[i] - d[i])

    peak_cap_b = rate
    peak_mid_b = float(dc_inv_mid_level.max()) if n else 0.0
    peak_eod_b = float(inv_beg_b[1:].max()) if n else 0.0

    # ========= Outputs (filenames kept EXACT) =========
    df_a = pd.DataFrame({
        "Date": daily["Date"],
        "Demand": d,
        "Target": t,
        "AF Production": af_prod_chase,
        "Strategy A Capacity Reserved": strategy_a_capacity_reserved,  # kept
        "DC Inventory": dc_inv_mid_chase,      # daily inventory series we plot
        "DC Inventory EOD": inv_beg[1:],
    })

    df_b = pd.DataFrame({
        "Date": daily["Date"],
        "Demand": d,
        "Target": t,
        "AF Steady Rate": rate,
        "DC Inventory": dc_inv_mid_level,      # daily inventory series we plot
        "DC Inventory EOD": inv_beg_b[1:],
    })

    df_sum = pd.DataFrame({
        "Peak Loads":   ["AF Peak Daily", "DC Peak", "DC Peak EOD"],
        "Follow Demand":[peak_cap_a,        peak_mid_a,  peak_eod_a],
        "Steady Rate":  [peak_cap_b,        peak_mid_b,  peak_eod_b],
    })

    # Filenames (including their original spellings)
    p_a = OUT_DIR / f"Task8 Strategy A {label}.csv"
    p_b = OUT_DIR / f"Task8 Startegy B {label}.csv"
    p_s = OUT_DIR / f"Task8 AF Startegy Summary {label}.csv"
    p_x = OUT_DIR / f"Task8 Report {label}.xlsx"

    df_a.to_csv(p_a, index=False)
    df_b.to_csv(p_b, index=False)
    df_sum.to_csv(p_s, index=False)

    with pd.ExcelWriter(p_x) as xw:
        df_a.to_excel(xw, sheet_name="Strategy A", index=False)
        df_b.to_excel(xw, sheet_name="Strategy B", index=False)
        df_sum.to_excel(xw, sheet_name="Summary", index=False)

    # ========= PNG charts (titles & names unchanged) =========
    # 1) AF production (daily)
    plt.figure(figsize=(11, 6))
    plt.plot(df_a["Date"], df_a["AF Production"], label="Strategy A (Chase)")
    plt.plot(df_b["Date"], df_b["AF Steady Rate"], label="Strategy B (Steady)")
    plt.plot(df_a["Date"], df_a["Strategy A Capacity Reserved"], label="Strategy A Capacity Reserved", linestyle="--")
    plt.xlabel("Date"); plt.ylabel("Daily AF Production")
    plt.title(f"Daily AF with Capacity — {label}")
    plt.legend(); plt.grid(True)
    png_daily = OUT_DIR / f"Daily AF {label}.png"
    plt.savefig(png_daily, bbox_inches="tight"); plt.close()

    # 2) AF production (cumulative)
    plt.figure(figsize=(11, 6))
    plt.plot(df_a["Date"], df_a["AF Production"].cumsum(), label="Strategy A (Chase) — Cumulative")
    plt.plot(df_b["Date"], df_b["AF Steady Rate"].cumsum(), label="Strategy B (Steady) — Cumulative")
    # cumulative of the reserved capacity is linear: peak_cap_a * day_index
    cum_reserved = np.arange(1, len(df_a) + 1, dtype=float) * df_a["Strategy A Capacity Reserved"].iloc[0]
    plt.plot(df_a["Date"], cum_reserved, label="Strategy A Capacity Reserved — Cumulative", linestyle="--")
    plt.xlabel("Date"); plt.ylabel("Cumulative AF Production")
    plt.title(f"Cumulative AF with Capacity — {label}")
    plt.legend(); plt.grid(True)
    png_cum = OUT_DIR / f"Cumulative AF {label}.png"
    plt.savefig(png_cum, bbox_inches="tight"); plt.close()

    # 3) Daily DC inventory (single series per strategy — no EOD vs mid split)
    plt.figure(figsize=(11, 6))
    plt.plot(df_a["Date"], df_a["DC Inventory"], label="Strategy A (Chase)")
    plt.plot(df_b["Date"], df_b["DC Inventory"], label="Strategy B (Steady)")
    plt.xlabel("Date"); plt.ylabel("DC Inventory (Units)")
    plt.title(f"Daily DC Inventory — {label}")
    plt.legend(); plt.grid(True)
    png_inv_daily = OUT_DIR / f"Daily DC Inventory {label}.png"
    plt.savefig(png_inv_daily, bbox_inches="tight"); plt.close()

    # 4) Cumulative DC inventory
    plt.figure(figsize=(11, 6))
    plt.plot(df_a["Date"], df_a["DC Inventory"].cumsum(), label="Strategy A (Chase) — Cumulative")
    plt.plot(df_b["Date"], df_b["DC Inventory"].cumsum(), label="Strategy B (Steady) — Cumulative")
    plt.xlabel("Date"); plt.ylabel("Cumulative DC Inventory (Units)")
    plt.title(f"Cumulative DC Inventory — {label}")
    plt.legend(); plt.grid(True)
    png_inv_cum = OUT_DIR / f"Cumulative DC Inventory {label}.png"
    plt.savefig(png_inv_cum, bbox_inches="tight"); plt.close()

    print(
        f"[Task8] ✅ {label} done → {OUT_DIR}\n"
        f"     PNG: {png_daily.name}, {png_cum.name}, {png_inv_daily.name}, {png_inv_cum.name}"
    )


# ========= RUN FOR 15FC, 4FC, 1FC (unchanged paths & semantics) =========
CONFIGS = {
    "15FC": Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7/FC Daily 15FC.csv"),
    "4FC":  Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7/FC Daily 4FC.csv"),
    "1FC":  Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7/FC Daily 1FC.csv"),
}

if __name__ == "__main__":
    try:
        # Keep the original behavior: aggregate at network level for 4FC and 15FC
        for label, path in CONFIGS.items():
            AGG_NET = label in ["4FC", "15FC"]
            run_task8(path, label)
        print("[Task8] 🎯 All configs done.")
    except Exception as e:
        print(f"[Task8] ERROR: {e}", file=sys.stderr)
        raise
