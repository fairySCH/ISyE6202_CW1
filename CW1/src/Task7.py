"""
Dobda Case — Task 7 Solution Script
Author: ChatGPT
Inputs (expected in CW1/):
  - data/zip3_pmf.csv
  - output/task1_2/df_global2_seasonality_daily.csv  (uses VAL_COL)
  - output/task3/assignment_15FC.csv
  - output/task3/assignment_4FC.csv
  - output/task3/assignment_1FC.csv

Outputs (written to CW1/output/task7/):
  - FC Daily {label}.csv
  - Network Daily {label}.csv
  - FC Max & Mean {label}.csv
  - Network Max {label}.csv
  - fc_target_max_{label}.png
  - dc_sumfc_total_{label}.png
  - network_service_levels_{label}_W{w}.png
  - network_service_level_delta_vs_99_{label}_W{w}.png
"""
import os
import sys
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Root folder for this casework and all input/output files
ROOT = "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1"

# Probability mass function for ZIP3 demand share
ZIP3_PMF = f"{ROOT}/data/zip3_pmf.csv"

# Daily demand profile (already seasonality-adjusted) with chosen value column
DAILY_CSV = f"{ROOT}/output/task1_2/df_global2_seasonality_daily.csv"
VAL_COL   = "units_2026_expected_distnormal_99"

# FC assignment files for the 3 configurations
ASSIGN = {
    "15FC": f"{ROOT}/output/task3/assignment_15FC.csv",
    "4FC":  f"{ROOT}/output/task3/assignment_4FC.csv",
    "1FC":  f"{ROOT}/output/task3/assignment_1FC.csv",
}

# All Task 7 outputs will be written here
OUT = f"{ROOT}/output/task7"
# ---------------------------------------------------------------

# Planning year and day level calender
YEAR = 2026
DATES = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")

# Service level Z-values and autonomy windows (days)
# Z_FC, Z_NET = z-score used for FC-level and network-level safety calculations for various level of customer satisfcation
# FC_DAYS, NET_DAYS = rolling window sizes (autonomy days) for FC and Network
Z_FC, Z_NET = 2.33, 2.33
FC_DAYS, NET_DAYS = 21, 56

# Scenario alternatives for sensitivity plots
SCEN_WEEKS = [6, 12]
SCEN_Z = {"50%": 0.0, "68%": 1.0, "95%": 1.645}


# -------------------------------
# Small utilities
# -------------------------------
def _assert_cols(df: pd.DataFrame, needed: List[str], label: str) -> None:
    """Fail fast if the DataFrame is missing required columns."""
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{label}] missing columns: {missing}")


def _normalize_pmf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the PMF column sums to 1.0 (if positive),
    so the shares cleanly allocate the total daily units.
    """
    s = df["pmf"].sum()
    if s > 0 and not np.isclose(s, 1.0):
        df["pmf"] = df["pmf"] / s
    return df


def _sigma_window(arr: np.ndarray) -> float:
    """
    Sample standard deviation of a window (ddof=1).
    Returns 0.0 if the window is < 2 points.
    """
    n = int(arr.size)
    if n < 2:
        return 0.0
    return float(np.std(arr.astype(float, copy=False), ddof=1))


def _inv_window_sigma(arr: np.ndarray, z: float) -> Tuple[float, float, float]:
    """
    Given a window of daily demand:
      cycle  = sum of demand over the window
      sigma  = sample std dev of daily demand in the window
      safety = z * sigma * sqrt(n)   (approx safety for the n-day sum)
      target = cycle + safety
    Returns (cycle, safety, target).
    """
    n = int(arr.size)
    if n == 0:
        return 0.0, 0.0, 0.0
    arr = arr.astype(float, copy=False)
    cycle = float(arr.sum())
    sigma = _sigma_window(arr)
    safety = float(z * sigma * np.sqrt(n))
    target = cycle + safety
    return cycle, safety, target


# -------------------------------
# IO helpers
# -------------------------------
def load_zip3_pmf(path: str) -> pd.DataFrame:
    """
    Load ZIP3 PMF: columns zip3, pmf.
    - Lowercases headers
    - Pads zip3 to 3 digits
    - Coerces pmf to [0, ∞), then normalizes to sum to 1
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.lower()
    _assert_cols(df, ["zip3", "pmf"], "zip3_pmf")

    df["zip3"] = df["zip3"].astype(str).str.zfill(3)
    df["pmf"] = pd.to_numeric(df["pmf"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return _normalize_pmf(df)[["zip3", "pmf"]]


def load_daily_units(path: str, dates: pd.DatetimeIndex, col: str) -> pd.Series:
    """
    Load the chosen daily units series from the Task 1/2 CSV.
    - Requires a 'date' column
    - Reindexes to the full-year DATES, filling gaps with 0
    """
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.lower()
    if "date" not in raw.columns:
        raise RuntimeError("[daily] expected a 'date' column")

    col_lower = col.lower()
    if col_lower not in raw.columns:
        raise RuntimeError(f"[daily] missing column '{col}'")

    s = pd.Series(
        pd.to_numeric(raw[col_lower], errors="coerce").fillna(0.0).values,
        index=pd.to_datetime(raw["date"])
    )
    s = s.reindex(dates).fillna(0.0).clip(lower=0.0)
    return s


def load_assignment(path: str) -> pd.DataFrame:
    """
    Load the ZIP3 -> FC assignment mapping.
    - Accepts 'preferred_fc' as alias for 'fc_id'
    - Returns only columns ['zip3', 'fc_id'] with zip3 padded to 3 digits
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.lower()
    if "preferred_fc" in df.columns and "fc_id" not in df.columns:
        df = df.rename(columns={"preferred_fc": "fc_id"})
    _assert_cols(df, ["zip3", "fc_id"], "assignment")
    df["zip3"] = df["zip3"].astype(str).str.zfill(3)
    return df[["zip3", "fc_id"]]


# -------------------------------
# Plot annotation helpers
# -------------------------------
def _annotate_line(ax, x, y, label_freq=30, fmt="{:.0f}"):
    """Light line labeling to help read values along the time axis."""
    for i in range(0, len(x), label_freq):
        ax.text(x[i], y[i], fmt.format(y[i]), fontsize=7, ha="center", va="bottom")


def _annotate_bar(ax, rects, fmt="{:.0f}"):
    """Label bars with their heights."""
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                fmt.format(height), ha="center", va="bottom", fontsize=8)


# -------------------------------
# Compute one configuration
# -------------------------------
def run_one_config(label: str, asg_path: str, zpmf: pd.DataFrame, units: pd.Series) -> None:
    """
    Build daily FC targets and network/DC inventories for a configuration.

    Steps
    1) Split the global daily units to ZIP3 using PMF shares.
    2) Map ZIP3 to FCs and sum ZIP3 streams into FC daily series.
    3) For each FC and each day i:
         - compute rolling n=FC_DAYS window (i .. i+n-1)
         - cycle, safety(z), target = _inv_window_sigma(...)
    4) Network daily = sum of FC daily; compute network target over NET_DAYS window.
    5) DC Inventory = Network Target - Sum of FC Targets (never negative).
    6) Save detailed CSVs and a few plots, plus summary stats.
    """
    # Load mapping ZIP3->FC for this config (e.g., 15FC)
    asg = load_assignment(asg_path)

    # Only ZIP3s that appear in both PMF and assignment are used
    merged = zpmf.merge(asg, on="zip3", how="inner")
    if merged.empty:
        raise RuntimeError(f"[{label}] no overlap between PMF and assignment zip3s")

    # Build per-ZIP3 daily streams by multiplying global units by that ZIP3's share
    by_zip: Dict[str, np.ndarray] = {}
    for _, row in merged.iterrows():
        z = row["zip3"]
        by_zip[z] = units.values * float(row["pmf"])
    zip_daily = pd.DataFrame(by_zip, index=units.index)

    # Group ZIP3s by their assigned FC
    fc_map = merged.set_index("zip3")["fc_id"].to_dict()
    fc_cols: Dict[str, List[str]] = {}
    for z in zip_daily.columns:
        fc = fc_map[z]
        fc_cols.setdefault(fc, []).append(z)

    # Sum ZIP3 columns to get per-FC daily demand series
    fc_daily = pd.DataFrame(index=units.index)
    for fc, cols in sorted(fc_cols.items()):
        fc_daily[fc] = zip_daily[cols].sum(axis=1)

    # ---------- FC daily targets ----------
    # For each FC and day i, compute the 21-day (FC_DAYS) cycle+safety target
    fc_rows: List[Dict[str, object]] = []
    for fc in fc_daily.columns:
        arr = fc_daily[fc].to_numpy()
        for i, dt in enumerate(units.index):
            cyc, saf, tgt = _inv_window_sigma(arr[i:i + FC_DAYS], Z_FC)
            fc_rows.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "FC ID": fc,
                "FC Average": cyc,   # rolling window sum over FC_DAYS
                "FC Safety": saf,    # z * sigma * sqrt(window)
                "FC Target": tgt     # Average + Safety
            })
    df_fc = pd.DataFrame(fc_rows)

    # ---------- FC Targets + DC ----------
    # FC Targets is the sum of all FCs per day
    net_arr = fc_daily.sum(axis=1).to_numpy()

    # For each day i, compute network target over 56 days (NET_DAYS)
    # Then set DC inventory = Network Target - Sum(FC Targets) for that day
    net_rows: List[Dict[str, object]] = []
    for i, dt in enumerate(units.index):
        nc, ns, nt = _inv_window_sigma(net_arr[i:i + NET_DAYS], Z_NET)
        dstr = dt.strftime("%Y-%m-%d")
        fc_sum_today = float(df_fc.loc[df_fc["Date"] == dstr, "FC Target"].sum())
        dc_inv = max(0.0, nt - fc_sum_today)
        net_rows.append({
            "Date": dstr,
            "Network Average": nc,
            "Network Safety": ns,
            "Network Target": nt,
            "DC Inventory": dc_inv
        })
    df_net = pd.DataFrame(net_rows)

    # ---------- Summaries ----------
    # FC-level stats: peak and mean FC target per FC
    df_fc_stats = (
        df_fc.groupby("FC ID", as_index=False)["FC Target"]
        .agg(**{"FC Target Max": "max", "FC Target Mean": "mean"})
    )

    # Network/DC summary stats across the year
    df_net_stats = pd.DataFrame([{
        "Config": label,
        "Num FCs": fc_daily.shape[1],
        "Network Target Max": float(df_net["Network Target"].max()),
        "Network Target Mean": float(df_net["Network Target"].mean()),
        "DC Inventory Max": float(df_net["DC Inventory"].max()),
        "DC Inventory Mean": float(df_net["DC Inventory"].mean()),
    }])

    os.makedirs(OUT, exist_ok=True)

    def _save_csv(df: pd.DataFrame, name: str) -> None:
        """Write a CSV named '{name} {label}.csv' into OUT."""
        p = os.path.join(OUT, f"{name} {label}.csv")
        df.to_csv(p, index=False)

    # Detailed time series and summaries
    _save_csv(df_fc,       "FC Daily")
    _save_csv(df_net,      "Network Daily")
    _save_csv(df_fc_stats, "FC Max & Mean")
    _save_csv(df_net_stats,"Network Max")

    # ------Graph Plots ----------
    import matplotlib.pyplot as plt

    # Plot 1: Max FC Target by FC (bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_fc_stats["FC ID"], df_fc_stats["FC Target Max"])
    ax.set_ylabel("FC Target Max")
    ax.set_title(f"FC Target Max by FC ({label})")
    plt.xticks(rotation=90)
    _annotate_bar(ax, bars)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"fc_target_max_{label}.png"))
    plt.close()

    # Plot 2: DC vs Sum(FC Targets) vs Total System Inventory (line)
    # Build a daily series of Sum(FC Targets) to compare with Network/DC
    fc_total_daily = (
        df_fc.groupby("Date", as_index=False)["FC Target"]
        .sum()
        .rename(columns={"FC Target": "FC Total Target"})
    )
    merge_plot = df_net.merge(fc_total_daily, on="Date", how="left")
    merge_plot["Total System Inventory"] = merge_plot["FC Total Target"] + merge_plot["DC Inventory"]

    x_dates = pd.to_datetime(merge_plot["Date"])
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_dates, merge_plot["DC Inventory"], label="DC Inventory")
    ax.plot(x_dates, merge_plot["FC Total Target"], label="Sum of FC Targets")
    ax.plot(x_dates, merge_plot["Total System Inventory"], label="Total System Inventory")

    # Light value annotations every ~month for readability
    _annotate_line(ax, x_dates, merge_plot["DC Inventory"].values, label_freq=30)
    _annotate_line(ax, x_dates, merge_plot["FC Total Target"].values, label_freq=30)
    _annotate_line(ax, x_dates, merge_plot["Total System Inventory"].values, label_freq=30)

    ax.set_ylabel("Inventory")
    ax.set_title(f"DC vs Sum(FC) vs Total System — {label}")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"dc_sumfc_total_{label}.png"))
    plt.close()

    # Plot 3/4: Network targets under different service levels (50/68/95/99%)
    # and their gap vs 99% for different autonomy windows (6 and 12 weeks)
    levels_for_plot = [("50%", SCEN_Z["50%"]), ("68%", SCEN_Z["68%"]),
                       ("95%", SCEN_Z["95%"]), ("99%", 2.33)]
    for w in SCEN_WEEKS:
        nd = int(w * 7)

        # Build a DataFrame with network targets per service level for this window length
        data = {}
        for lvl, zval in levels_for_plot:
            vals = []
            for i in range(len(units.index)):
                _, _, nt = _inv_window_sigma(net_arr[i:i + nd], float(zval))
                vals.append(nt)
            data[lvl] = vals

        df_levels = pd.DataFrame(data, index=units.index)

        # Plot A: absolute network target for each service level
        fig, ax = plt.subplots(figsize=(12, 7))
        for lvl in ["50%", "68%", "95%", "99%"]:
            ax.plot(df_levels.index, df_levels[lvl], label=lvl)
            _annotate_line(ax, df_levels.index, df_levels[lvl].values, label_freq=56)
        ax.set_ylabel("Network Target Inventory")
        ax.set_title(f"Network Service Levels — {label} W{w}")
        ax.legend(title="Service Level")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"network_service_levels_{label}_W{w}.png"))
        plt.close()

        # Plot B: inventory gap vs the 99% target (how much buffer you'd save if you accept lower SL)
        fig, ax = plt.subplots(figsize=(12, 7))
        for lvl in ["50%", "68%", "95%"]:
            delta = df_levels["99%"] - df_levels[lvl]
            ax.plot(df_levels.index, delta, label=f"{lvl} Δ to 99%")
            _annotate_line(ax, df_levels.index, delta.values, label_freq=56)
        ax.set_ylabel("Inventory Gap vs 99%")
        ax.set_title(f"Service Level Δ vs 99% — {label} W{w}")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"network_service_level_delta_vs_99_{label}_W{w}.png"))
        plt.close()

    print(f"[Task7] {label}: wrote outputs to {OUT}")


def main() -> None:
    """
    Orchestrates the three configurations (15FC/4FC/1FC):
      - Loads PMF and daily units
      - Runs the pipeline per config
      - Writes CSVs and plots into OUT
    """
    os.makedirs(OUT, exist_ok=True)

    # Load demand share (ZIP3 PMF) and daily units for the year
    zpmf  = load_zip3_pmf(ZIP3_PMF)
    units = load_daily_units(DAILY_CSV, DATES, VAL_COL)

    if units.empty:
        raise RuntimeError("[Task7] daily units came back empty")
    if not np.isfinite(units.values).all():
        raise RuntimeError("[Task7] non-finite entries in daily units")

    # Run for each FC configuration
    for label, path in ASSIGN.items():
        run_one_config(label, path, zpmf, units)

    print("[Task7] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print a concise error to stderr and re-raise for full trace in logs
        print(f"[Task7] ERROR: {e}", file=sys.stderr)
        raise
