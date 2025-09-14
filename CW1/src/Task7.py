
"""
Task 7 — FC & Network Inventory Targets + Scenario Plots

Refactor notes (no behavior change):
- Output CSV/Excel filenames and PNG filenames are unchanged.
- Plot titles/text are unchanged.
- Core math is untouched; only code organization, comments, and small guardrails improved.
"""

import os
import sys
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# --- EDIT THESE PATHS (kept exactly as your original constants) ---
ROOT = "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1"
ZIP3_PMF = f"{ROOT}/data/zip3_pmf.csv"
DAILY_CSV = f"{ROOT}/output/task1_2/df_global2_seasonality_daily.csv"
VAL_COL   = "units_2026_expected_distnormal_99"
ASSIGN = {
    "15FC": f"{ROOT}/output/task3/assignment_15FC.csv",
    "4FC":  f"{ROOT}/output/task3/assignment_4FC.csv",
    "1FC":  f"{ROOT}/output/task3/assignment_1FC.csv",
}
OUT = f"{ROOT}/output/task7"
# ---------------------------------------------------------------

YEAR = 2026
DATES = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")

# 99% service level → ~2.33 z-score; FC horizon 3 weeks (21d), Network 8 weeks (56d)
CV, Z_FC, Z_NET = 0.15, 2.33, 2.33
FC_DAYS, NET_DAYS = 21, 56

# Scenario alternatives (unchanged)
SCEN_WEEKS = [6, 12]
SCEN_Z = {"50%": 0.0, "68%": 1.0, "95%": 1.645}


# -------------------------------
# Small utilities
# -------------------------------
def _assert_cols(df: pd.DataFrame, needed: List[str], label: str) -> None:
    """Raise if required columns are missing."""
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{label}] missing columns: {missing}")


def _normalize_pmf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize PMF to sum to 1 if positive and not already ~1.
    In-place is fine here; returns the same frame for chaining.
    """
    s = df["pmf"].sum()
    if s > 0 and not np.isclose(s, 1.0):
        df["pmf"] = df["pmf"] / s
    return df


def _inv_window(arr: np.ndarray, cv: float, z: float) -> Tuple[float, float, float]:
    """
    Rolling inventory window for a demand slice `arr`:
      cycle = sum(arr)
      safety = z * sqrt(sum((cv * arr)^2))
      target = cycle + safety
    """
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    arr = arr.astype(float, copy=False)
    cyc = float(arr.sum())
    saf = float(z * np.sqrt(np.sum((cv * arr) ** 2)))
    return cyc, saf, cyc + saf


# -------------------------------
# IO helpers
# -------------------------------
def load_zip3_pmf(path: str) -> pd.DataFrame:
    """
    Read zip3 PMF, normalize, and return columns: ['zip3','pmf'].
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.lower()
    _assert_cols(df, ["zip3", "pmf"], "zip3_pmf")

    df["zip3"] = df["zip3"].astype(str).str.zfill(3)
    df["pmf"] = pd.to_numeric(df["pmf"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return _normalize_pmf(df)[["zip3", "pmf"]]


def load_daily_units(path: str, dates: pd.DatetimeIndex, col: str) -> pd.Series:
    """
    Load daily series for the full year index `dates`.
    Holes are zero-filled and negatives are clamped to 0.
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
    # force full-year daily index; any holes become 0
    s = s.reindex(dates).fillna(0.0).clip(lower=0.0)
    return s


def load_assignment(path: str) -> pd.DataFrame:
    """
    Read zip3→FC assignment. Accepts 'preferred_fc' in place of 'fc_id'.
    Returns ['zip3','fc_id'] with zero-padded zip3.
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.lower()

    # tolerate common naming variants
    if "preferred_fc" in df.columns and "fc_id" not in df.columns:
        df = df.rename(columns={"preferred_fc": "fc_id"})

    _assert_cols(df, ["zip3", "fc_id"], "assignment")

    df["zip3"] = df["zip3"].astype(str).str.zfill(3)
    return df[["zip3", "fc_id"]]


# -------------------------------
# Compute one configuration (1FC/4FC/15FC)
# -------------------------------
def run_one_config(label: str, asg_path: str, zpmf: pd.DataFrame, units: pd.Series) -> None:
    """
    Build per-FC daily targets and network/DC metrics for one assignment configuration.
    Saves CSVs, an Excel bundle, and required plots. Filenames unchanged.
    """
    # Join PMF with assignment; if empty, nothing to do for this config.
    asg = load_assignment(asg_path)
    merged = zpmf.merge(asg, on="zip3", how="inner")
    if merged.empty:
        raise RuntimeError(f"[{label}] no overlap between PMF and assignment zip3s")

    # zip3 daily demand = global units * pmf
    by_zip: Dict[str, np.ndarray] = {}
    for _, row in merged.iterrows():
        z = row["zip3"]
        by_zip[z] = units.values * float(row["pmf"])
    zip_daily = pd.DataFrame(by_zip, index=units.index)

    # Group zip3 → FC (build index of columns for each FC)
    fc_map = merged.set_index("zip3")["fc_id"].to_dict()
    fc_cols: Dict[str, List[str]] = {}
    for z in zip_daily.columns:
        fc = fc_map[z]
        fc_cols.setdefault(fc, []).append(z)

    # Aggregate zip3 columns per FC
    fc_daily = pd.DataFrame(index=units.index)
    for fc, cols in sorted(fc_cols.items()):
        fc_daily[fc] = zip_daily[cols].sum(axis=1)

    # ---------------------------
    # FC daily targets (3 weeks @ 99%)
    # ---------------------------
    fc_rows: List[Dict[str, object]] = []
    arr_by_fc = {fc: fc_daily[fc].to_numpy() for fc in fc_daily.columns}

    for fc, arr in arr_by_fc.items():
        # rolling window per day
        for i, dt in enumerate(units.index):
            cyc, saf, tgt = _inv_window(arr[i:i + FC_DAYS], CV, Z_FC)
            fc_rows.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "FC ID": fc,
                "FC Average": cyc,
                "FC Safety": saf,
                "FC Target": tgt
            })

    df_fc = pd.DataFrame(fc_rows)

    # ---------------------------
    # Network + DC (8 weeks @ 99%)
    # ---------------------------
    net_arr = fc_daily.sum(axis=1).to_numpy()
    net_rows: List[Dict[str, object]] = []

    for i, dt in enumerate(units.index):
        nc, ns, nt = _inv_window(net_arr[i:i + NET_DAYS], CV, Z_NET)
        dstr = dt.strftime("%Y-%m-%d")

        # Sum of FC targets for "today" (to compute DC inventory that sits above FCs)
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

    # ---------------------------
    # Summaries (per-FC and network/DC)
    # ---------------------------
    df_fc_stats = (
        df_fc.groupby("FC ID", as_index=False)["FC Target"]
        .agg(**{"FC Target Max": "max", "FC Target Mean": "mean"})
    )
    df_net_stats = pd.DataFrame([{
        "Config": label,
        "Num FCs": fc_daily.shape[1],
        "Network Target Max": float(df_net["Network Target"].max()),
        "Network Target Mean": float(df_net["Network Target"].mean()),
        "DC Inventory Max": float(df_net["DC Inventory"].max()),
        "DC Inventory Mean": float(df_net["DC Inventory"].mean()),
    }])

    # ---------------------------
    # Scenario alternatives (6/12 weeks × 50/68/95)
    # ---------------------------
    scen_rows: List[Dict[str, object]] = []
    for w in SCEN_WEEKS:
        nd = int(w * 7)
        for lvl, zval in SCEN_Z.items():
            for i, dt in enumerate(units.index):
                na, ns, nt = _inv_window(net_arr[i:i + nd], CV, float(zval))
                dstr = dt.strftime("%Y-%m-%d")
                fc_sum_today = float(df_fc.loc[df_fc["Date"] == dstr, "FC Target"].sum())
                scen_rows.append({
                    "Config": label,
                    "Date": dstr,
                    "Weeks": w,
                    "Service Level": lvl,
                    "Network Average": na,
                    "Network Safety": ns,
                    "Network Target": nt,
                    "FC Total Target Today": fc_sum_today,
                    "DC Inventory": max(0.0, nt - fc_sum_today),
                })

    df_scen = pd.DataFrame(scen_rows)
    df_scen_sum = (
        df_scen.groupby(["Config", "Weeks", "Service Level"], as_index=False)
        .agg(**{
            "Network Max Target": ("Network Target", "max"),
            "Network Mean Target": ("Network Target", "mean"),
            "DC Inventory Max": ("DC Inventory", "max"),
            "DC Inventory Mean": ("DC Inventory", "mean"),
        })
    )

    # === SAVE with the exact same filenames ===
    os.makedirs(OUT, exist_ok=True)

    def _save_csv(df: pd.DataFrame, name: str) -> None:
        p = os.path.join(OUT, f"{name} {label}.csv")
        df.to_csv(p, index=False)

    _save_csv(df_fc,       "FC Daily")
    _save_csv(df_net,      "Network Daily")
    _save_csv(df_fc_stats, "FC Max & Mean")
    _save_csv(df_net_stats,"Network Max")
    _save_csv(df_scen,     "Scenarios Daily")
    _save_csv(df_scen_sum, "Scenarios Summary")

    # Excel bundle (same sheet names, unchanged filename)
    with pd.ExcelWriter(os.path.join(OUT, f"Task7 Report {label}.xlsx")) as xw:
        df_fc.to_excel(xw, "FC Daily", index=False)
        df_net.to_excel(xw, "Network Daily", index=False)
        df_fc_stats.to_excel(xw, "FC Max & Mean", index=False)
        df_net_stats.to_excel(xw, "Network Max", index=False)
        df_scen.to_excel(xw, "Scenarios Daily", index=False)
        df_scen_sum.to_excel(xw, "Scenarios Summary", index=False)

    # ---------------------------
    # PLOTS (per-config) — filenames & titles are IDENTICAL
    # ---------------------------
    import matplotlib.pyplot as plt  # local import keeps global namespace tidy

    # 1) FC Target Max by FC (bar)
    plt.figure(figsize=(10, 6))
    plt.bar(df_fc_stats["FC ID"], df_fc_stats["FC Target Max"])
    plt.xticks(rotation=90)
    plt.ylabel("FC Target Max")
    plt.title(f"FC Target Max by FC ({label})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"fc_target_max_{label}.png"))
    plt.close()

    # 2) DC vs Sum(FC Targets) vs Total System
    fc_total_daily = (
        df_fc.groupby("Date", as_index=False)["FC Target"]
        .sum()
        .rename(columns={"FC Target": "FC Total Target"})
    )
    merge_plot = df_net.merge(fc_total_daily, on="Date", how="left")
    merge_plot["Total System Inventory"] = merge_plot["FC Total Target"] + merge_plot["DC Inventory"]

    x_dates = pd.to_datetime(merge_plot["Date"])

    plt.figure(figsize=(12, 7))
    plt.plot(x_dates, merge_plot["DC Inventory"], label="DC Inventory")
    plt.plot(x_dates, merge_plot["FC Total Target"], label="Sum of FC Targets")
    plt.plot(x_dates, merge_plot["Total System Inventory"], label="Total System Inventory")
    plt.xticks(rotation=45)
    plt.ylabel("Inventory")
    plt.title(f"DC vs Sum(FC) vs Total System — {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"dc_sumfc_total_{label}.png"))
    plt.close()

    # 3) Service level values (50/68/95/99) and Difference to 99%
    levels_for_plot = [("50%", SCEN_Z["50%"]), ("68%", SCEN_Z["68%"]),
                       ("95%", SCEN_Z["95%"]), ("99%", 2.33)]

    for w in SCEN_WEEKS:
        nd = int(w * 7)

        # Build per-level Network Target time series
        data = {}
        for lvl, zval in levels_for_plot:
            vals = []
            for i in range(len(units.index)):
                _, _, nt = _inv_window(net_arr[i:i + nd], CV, float(zval))
                vals.append(nt)
            data[lvl] = vals

        df_levels = pd.DataFrame(data, index=units.index)

        # Plot A: Service level values (50/68/95/99)
        plt.figure(figsize=(12, 7))
        for lvl in ["50%", "68%", "95%", "99%"]:
            plt.plot(df_levels.index, df_levels[lvl], label=lvl)
        plt.xticks(rotation=45)
        plt.ylabel("Network Target Inventory")
        plt.title(f"Network Service Levels — {label} W{w}")
        plt.legend(title="Service Level")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"network_service_levels_{label}_W{w}.png"))
        plt.close()

        # Plot B: Difference to 99% (how much lower 50/68/95 are vs 99)
        plt.figure(figsize=(12, 7))
        for lvl in ["50%", "68%", "95%"]:
            delta = df_levels["99%"] - df_levels[lvl]
            plt.plot(df_levels.index, delta, label=f"{lvl} Δ to 99%")
        plt.xticks(rotation=45)
        plt.ylabel("Inventory Gap vs 99%")
        plt.title(f"Service Level Δ vs 99% — {label} W{w}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"network_service_level_delta_vs_99_{label}_W{w}.png"))
        plt.close()

    print(f"[Task7] {label}: wrote outputs to {OUT}")


def main() -> None:
    """Entry point: load inputs and run all configurations."""
    os.makedirs(OUT, exist_ok=True)

    # Load inputs (unchanged)
    zpmf  = load_zip3_pmf(ZIP3_PMF)
    units = load_daily_units(DAILY_CSV, DATES, VAL_COL)

    # Light sanity checks
    if units.empty:
        raise RuntimeError("[Task7] daily units came back empty")
    if not np.isfinite(units.values).all():
        raise RuntimeError("[Task7] non-finite entries in daily units")

    # Run per config
    for label, path in ASSIGN.items():
        run_one_config(label, path, zpmf, units)

    # NOTE: previously-added combined plot network_fc_dc_ALL is intentionally not present.
    print("[Task7] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Task7] ERROR: {e}", file=sys.stderr)
        raise
