# Task7_fixed_v2.py
import os
import glob
import numpy as np
import pandas as pd
from math import ceil

# ---------------------------
# PARAMETERS (adjustable)
# ---------------------------
ROBUSTNESS = {0.50: 0.0, 0.68: 0.468, 0.95: 1.645, 0.99: 2.33}
FC_AUTONOMY_DAYS = 21         # 3 weeks
NETWORK_AUTONOMY_DAYS = 56    # 8 weeks
CV_DAILY = 0.15               # coefficient of variation for per-day demand uncertainty
YEAR = 2026
DATES = pd.date_range(start=f"{YEAR}-01-01", end=f"{YEAR}-12-31", freq='D')

# FILE PATHS - update if needed
paths = {
    # original file you pointed to (may be aggregate): script will try to find a better file if needed
    "fc_zip3_assignment_with_pmf": r"D:\ISyE6202_CW1\CW1\data\dobda_task3_outputs\task3c_fc_market_distance_distribution_15FC.csv",
    "demand_seasonalities": r"D:\ISyE6202_CW1\CW1\data\dobda_demand_seasonalities.csv"  # optional
}

# Market parameters (median scenario)
TOTAL_MARKET_TODAY = 2_000_000
CURRENT_MARKET_SHARE = 0.036
OVERALL_MARKET_GROW_MEDIAN = 0.075
DODBA_MARKET_SHARE_GROW_MEDIAN = 0.20

# ---------------------------
# Helper utilities
# ---------------------------
def check_file(path, friendly=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}" + (f" ({friendly})" if friendly else ""))
    return path

def normalize_headers(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def try_find_zip3_file(folder):
    """Look for any CSV in folder that contains a 'zip3' column; return path or None."""
    for f in glob.glob(os.path.join(folder, "*.csv")):
        try:
            tmp = pd.read_csv(f, nrows=2, dtype=str)
            tmp.columns = [c.strip().lower() for c in tmp.columns]
            if "zip3" in tmp.columns:
                return f
        except Exception:
            continue
    return None

def load_fc_zip3_with_pmf(path):
    """
    Robust loader:
    - expects a CSV containing at least zip3, preferred_fc (or fc_id), and a pmf-like column.
    - if the provided file lacks zip3, attempts to find any CSV in same folder that has zip3.
    """
    # ensure file exists
    path = check_file(path, "fc_zip3_assignment_with_pmf")
    folder = os.path.dirname(path)

    # load the provided file first
    df = pd.read_csv(path, dtype=str)
    df = normalize_headers(df)
    print("DEBUG: columns in provided assignment file:", df.columns.tolist())

    # If 'zip3' not present, try to locate another CSV in same folder that has 'zip3'
    if "zip3" not in df.columns:
        alt = try_find_zip3_file(folder)
        if alt:
            print(f"INFO: Provided file lacks 'zip3'. Found alternative file with zip3: {alt}")
            df = pd.read_csv(alt, dtype=str)
            df = normalize_headers(df)
            print("DEBUG: columns in alternative file:", df.columns.tolist())
        else:
            # nothing usable found — show helpful message and raise
            raise KeyError(
                "Assignment file does not contain 'zip3'. "
                f"Available columns in provided file: {df.columns.tolist()}. "
                "Please supply the Task-3 ZIP3-level assignment file (columns: zip3, preferred_fc (or fc_id), pmf/demand_share)."
            )

    # rename preferred_fc -> fc_id if present
    if "preferred_fc" in df.columns and "fc_id" not in df.columns:
        df = df.rename(columns={"preferred_fc": "fc_id"})

    # detect pmf-like column automatically
    pmf_col = None
    for candidate in ("pmf", "demand_share", "pmf_norm", "pmf_normalized", "demand_share_norm"):
        if candidate in df.columns:
            pmf_col = candidate
            break

    if "zip3" not in df.columns or "fc_id" not in df.columns or pmf_col is None:
        # provide very explicit debugging info
        raise KeyError(
            "Assignment file must contain 'zip3', 'preferred_fc' (or 'fc_id'), and a pmf-like column.\n"
            f"Columns found: {df.columns.tolist()}\n"
            "If your file uses different names, either rename the header row or provide the correct Task-3 ZIP3 assignment file."
        )

    # normalize zip3 format, ensure pmf numeric
    df = df[["zip3", "fc_id", pmf_col]].rename(columns={pmf_col: "pmf"})
    df["zip3"] = df["zip3"].astype(str).str.zfill(3)
    df["pmf"] = df["pmf"].astype(float)

    return df

def compute_annual_dobda_units():
    overall_next = TOTAL_MARKET_TODAY * (1 + OVERALL_MARKET_GROW_MEDIAN)
    dodba_share_next = CURRENT_MARKET_SHARE * (1 + DODBA_MARKET_SHARE_GROW_MEDIAN)
    return overall_next * dodba_share_next

def load_seasonality_if_available(path):
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = normalize_headers(df)
    week_map = None
    dow_map = None
    if {"week", "factor"}.issubset(set(df.columns)):
        week_map = df.set_index("week")["factor"].to_dict()
    if {"dow", "factor"}.issubset(set(df.columns)) or {"day_of_week", "factor"}.issubset(set(df.columns)):
        if "dow" in df.columns:
            dow_map = df.set_index("dow")["factor"].to_dict()
        else:
            dow_map = df.set_index("day_of_week")["factor"].to_dict()
    return {"week": week_map, "dow": dow_map}

def build_day_factors(dates, seasonality=None):
    df = pd.DataFrame({"date": dates})
    df["week"] = df["date"].dt.isocalendar().week
    df["dow"] = df["date"].dt.dayofweek
    if seasonality is None or (seasonality.get("week") is None and seasonality.get("dow") is None):
        df["factor"] = 1.0
    else:
        week_map = seasonality.get("week") or {}
        dow_map = seasonality.get("dow") or {}
        df["week_factor"] = df["week"].map(week_map).fillna(1.0)
        df["dow_factor"] = df["dow"].map(dow_map).fillna(1.0)
        df["factor"] = df["week_factor"] * df["dow_factor"]
    df["factor_norm"] = df["factor"] / df["factor"].sum()
    return pd.Series(df["factor_norm"].values, index=df["date"], name="day_factor")

def produce_zip3_daily(zip3_pmf_df, day_factors):
    dates = day_factors.index
    mat = {}
    for _, row in zip3_pmf_df.iterrows():
        zip3 = str(row["zip3"]).zfill(3)
        yearly = row["yearly_demand"]
        mat[zip3] = day_factors.values * yearly
    return pd.DataFrame(mat, index=dates)

def aggregate_zip3_to_fc(zip3_daily_df, fc_map_df):
    mapping = fc_map_df.set_index("zip3")["fc_id"].to_dict()
    missing = [z for z in zip3_daily_df.columns if z not in mapping]
    if missing:
        print(f"⚠️ Warning: {len(missing)} zip3 codes missing assignment; dropping example: {missing[:6]}")
        zip3_daily_df = zip3_daily_df.drop(columns=missing)
    fc_cols = {}
    for zip3 in zip3_daily_df.columns:
        fc = mapping.get(zip3)
        if fc:
            fc_cols.setdefault(fc, []).append(zip3)
    fc_daily = {fc: zip3_daily_df[cols].sum(axis=1).values for fc, cols in fc_cols.items()}
    return pd.DataFrame(fc_daily, index=zip3_daily_df.index)

def daily_target_from_future_slice(future_expected_array, cv=CV_DAILY, z=ROBUSTNESS[0.99]):
    arr = np.array(future_expected_array, dtype=float)
    cycle = float(arr.sum())
    var_total = float(np.sum((cv * arr) ** 2))
    safety = float(z * np.sqrt(var_total))
    return cycle, safety, cycle + safety

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    # Load assignment+pmf file (robust)
    assignment_df = load_fc_zip3_with_pmf(paths["fc_zip3_assignment_with_pmf"])

    # Build zip3 pmf DF using that file
    zip3_pmf_df = assignment_df[["zip3", "pmf"]].copy()

    # Compute annual Dobda demand
    annual_units = compute_annual_dobda_units()
    print(f"Annual Dobda demand (median scenario): {int(annual_units):,} units")

    # Attach yearly_demand per zip3
    zip3_pmf_df["yearly_demand"] = zip3_pmf_df["pmf"] * annual_units

    # Build day factors (seasonality optional)
    seasonality = load_seasonality_if_available(paths.get("demand_seasonalities"))
    day_factors = build_day_factors(DATES, seasonality=seasonality)

    # Produce zip3 x daily matrix
    zip3_daily_df = produce_zip3_daily(zip3_pmf_df, day_factors)

    # Aggregate to FC daily series
    fc_daily_df = aggregate_zip3_to_fc(zip3_daily_df, assignment_df[["zip3", "fc_id"]])

    if fc_daily_df.shape[1] == 0:
        raise RuntimeError("No FC columns produced. Check assignment file mappings.")

    # Compute FC daily targets (3-week, 99%)
    fc_daily_targets = []
    for fc in fc_daily_df.columns:
        arr = fc_daily_df[fc].values
        for t in range(len(arr)):
            future = arr[t: t + FC_AUTONOMY_DAYS]
            if future.size == 0:
                c = s = total = 0.0
            else:
                c, s, total = daily_target_from_future_slice(future, cv=CV_DAILY, z=ROBUSTNESS[0.99])
            fc_daily_targets.append({
                "date": fc_daily_df.index[t].strftime("%Y-%m-%d"),
                "fc_id": fc,
                "fc_cycle": c,
                "fc_safety": s,
                "fc_target": total
            })
    fc_daily_targets_df = pd.DataFrame(fc_daily_targets)

    # Compute network daily totals (8-week, 99%) and DC complement
    arr_network = fc_daily_df.sum(axis=1).values
    network_records = []
    for t in range(len(arr_network)):
        future_net = arr_network[t: t + NETWORK_AUTONOMY_DAYS]
        if future_net.size == 0:
            net_c = net_s = net_total = dc_t = 0.0
        else:
            net_c, net_s, net_total = daily_target_from_future_slice(future_net, cv=CV_DAILY, z=ROBUSTNESS[0.99])
            date_str = fc_daily_df.index[t].strftime("%Y-%m-%d")
            fc_targets_today = fc_daily_targets_df.loc[fc_daily_targets_df["date"] == date_str, "fc_target"].sum()
            dc_t = max(0.0, net_total - fc_targets_today)
        network_records.append({
            "date": fc_daily_df.index[t].strftime("%Y-%m-%d"),
            "network_cycle": net_c,
            "network_safety": net_s,
            "network_target": net_total,
            "dc_inventory": dc_t
        })
    network_daily_df = pd.DataFrame(network_records)

    # Maxima
    fc_max = (fc_daily_targets_df.groupby("fc_id")["fc_target"]
              .agg(["max", "mean"]).reset_index().rename(columns={"max": "fc_target_max", "mean": "fc_target_mean"}))
    network_max = {
        "network_target_max": float(network_daily_df["network_target"].max()),
        "network_target_mean": float(network_daily_df["network_target"].mean()),
        "dc_inventory_max": float(network_daily_df["dc_inventory"].max()),
        "dc_inventory_mean": float(network_daily_df["dc_inventory"].mean())
    }

    # Alternative scenarios: weeks [6,12] and robustness [50%,68%,95%]
    alt_results = []
    alt_weeks = [6, 12]
    alt_robs = [0.50, 0.68, 0.95]
    for weeks in alt_weeks:
        Nd = weeks * 7
        for conf in alt_robs:
            z = ROBUSTNESS[conf]
            net_totals = []
            for t in range(len(arr_network)):
                fut = arr_network[t: t + Nd]
                if fut.size == 0:
                    net_totals.append(0.0)
                else:
                    _, _, tot = daily_target_from_future_slice(fut, cv=CV_DAILY, z=z)
                    net_totals.append(tot)
            alt_results.append({
                "weeks": weeks,
                "confidence": conf,
                "network_max_target": float(np.max(net_totals)),
                "network_mean_target": float(np.mean(net_totals))
            })
    alt_results_df = pd.DataFrame(alt_results)

    # Save outputs
    out_dir = os.getcwd()
    fc_daily_targets_df.to_csv(os.path.join(out_dir, "task7_fc_daily_targets.csv"), index=False)
    network_daily_df.to_csv(os.path.join(out_dir, "task7_network_daily.csv"), index=False)
    fc_max.to_csv(os.path.join(out_dir, "task7_fc_maximums.csv"), index=False)
    pd.DataFrame([network_max]).to_csv(os.path.join(out_dir, "task7_network_maximums.csv"), index=False)
    alt_results_df.to_csv(os.path.join(out_dir, "task7_scenarios_summary.csv"), index=False)

    # Print short summary
    print("\nSaved outputs to:", out_dir)
    print(" - task7_fc_daily_targets.csv (per-FC daily cycle/safety/target)")
    print(" - task7_network_daily.csv (network daily cycle/safety/target + DC complement)")
    print(" - task7_fc_maximums.csv (max/mean per FC target)")
    print(" - task7_network_maximums.csv (network maxima/means)")
    print(" - task7_scenarios_summary.csv (alternative weeks/confidence)")
    print("\nExample FC maxima (top 8):")
    print(fc_max.head(8).to_string(index=False))
    print("\nNetwork maxima summary:")
    print(network_max)

if __name__ == "__main__":
    main()
