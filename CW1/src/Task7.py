# Task7.py — uses real daily demand from CSV (no normalization), builds FC & Network inventories,
# DC complement, and daily scenarios for 6/12 weeks × 50/68/95%.

import os
import numpy as np
import pandas as pd
from math import sqrt

# ========= CONFIG (update these 3 paths if needed) =========
PATHS = {
    "zip3_pmf": "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/data/zip3_pmf.csv",
    "fc_assignment": "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task3/assignment_15FC.csv",
    "daily_profile_csv": "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task1_2/df_global2_seasonality_daily.csv",
    # column in daily profile to use (must exist in the CSV; no normalization will be applied)
    "daily_profile_value_col": "units_2026_expected_distnormal_99",
}

OUTPUT_DIR = "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7"

# ========= PARAMETERS =========
YEAR = 2026
DATES = pd.date_range(start=f"{YEAR}-01-01", end=f"{YEAR}-12-31", freq="D")

# Service levels (z-scores)
Z_BASELINE_FC = 2.33     # 99% for FC
Z_BASELINE_NET = 2.33    # 99% for Network
CV_DAILY = 0.15          # coefficient of variation per day (assumed independent by day)

FC_AUTONOMY_DAYS = 21    # 3 weeks, FC baseline
NET_AUTONOMY_DAYS = 56   # 8 weeks, Network baseline

# Scenario service levels for 7c (use ~1.0 for 68%)
SCENARIO_Z = {"50%": 0.0, "68%": 1.0, "95%": 1.645}
SCENARIO_WEEKS = [6, 12]

# ========= HELPERS =========
def check_file(path, label=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}" + (f" ({label})" if label else ""))
    return path

def normalize_headers(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_zip3_pmf(path):
    path = check_file(path, "zip3 PMF")
    df = pd.read_csv(path, dtype=str)
    df = normalize_headers(df)
    if "zip3" not in df.columns or "pmf" not in df.columns:
        raise KeyError("zip3_pmf.csv must have columns: zip3, pmf")
    df["zip3"] = df["zip3"].astype(str).str.zfill(3)
    df["pmf"] = df["pmf"].astype(float)
    s = df["pmf"].sum()
    if not np.isclose(s, 1.0):
        # keep behaviour predictable: renormalize but warn
        print(f"[WARN] zip3 PMF sums to {s:.6f} — renormalizing to 1.0")
        df["pmf"] = df["pmf"] / s if s > 0 else df["pmf"]
    return df[["zip3", "pmf"]]

def load_fc_assignment(path):
    path = check_file(path, "FC assignment")
    df = pd.read_csv(path, dtype=str)
    df = normalize_headers(df)
    # allow preferred_fc or fc_id
    if "preferred_fc" in df.columns and "fc_id" not in df.columns:
        df = df.rename(columns={"preferred_fc": "fc_id"})
    if "zip3" not in df.columns or "fc_id" not in df.columns:
        raise KeyError("assignment file must have columns: zip3, fc_id (or preferred_fc)")
    df["zip3"] = df["zip3"].astype(str).str.zfill(3)
    return df[["zip3", "fc_id"]]

def load_daily_units(csv_path, dates, value_col):
    """
    Use the exact daily units provided in your CSV (no normalization/smoothing).
    Missing dates are filled with 0.0.
    """
    path = check_file(csv_path, "daily demand profile")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise KeyError("daily profile CSV must have a 'date' column")
    if value_col not in df.columns:
        raise KeyError(f"'{value_col}' not found in the daily profile CSV")
    df["date"] = pd.to_datetime(df["date"])
    s = pd.Series(df[value_col].astype(float).values, index=df["date"])
    s = s.reindex(dates).fillna(0.0)
    return pd.Series(s.values, index=dates, name="daily_units")

def split_to_zip3_daily(daily_units, zip3_pmf_df):
    """
    For each zip3, daily demand = daily_units * pmf_zip3 (so sum over zip3 == daily_units).
    Returns DataFrame indexed by date, columns = zip3.
    """
    mat = {}
    for _, row in zip3_pmf_df.iterrows():
        z3 = row["zip3"]
        share = float(row["pmf"])
        mat[z3] = daily_units.values * share
    return pd.DataFrame(mat, index=daily_units.index)

def aggregate_zip3_to_fc(zip3_daily_df, fc_assignment_df):
    """
    Sum zip3 daily columns to their assigned FC.
    """
    mapping = fc_assignment_df.set_index("zip3")["fc_id"].to_dict()
    # drop any zip3 not in mapping
    missing = [c for c in zip3_daily_df.columns if c not in mapping]
    if missing:
        print(f"[WARN] {len(missing)} ZIP3 missing FC mapping — dropping some columns (e.g. {missing[:6]})")
        zip3_daily_df = zip3_daily_df.drop(columns=missing)
    fc_cols = {}
    for z3 in zip3_daily_df.columns:
        fc = mapping.get(z3)
        if fc:
            fc_cols.setdefault(fc, []).append(z3)
    fc_daily = {fc: zip3_daily_df[zcols].sum(axis=1).values for fc, zcols in fc_cols.items()}
    return pd.DataFrame(fc_daily, index=zip3_daily_df.index).sort_index(axis=1)

def inv_from_future_slice(future_means, cv, z):
    """
    Given an array of expected daily means over the autonomy window:
      cycle = sum(means)
      safety = z * sqrt(sum((cv * mean)^2))
      target = cycle + safety
    """
    arr = np.asarray(future_means, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    cycle = float(arr.sum())
    var_total = float(np.sum((cv * arr) ** 2))
    safety = float(z * sqrt(var_total))
    return cycle, safety, cycle + safety

# ========= MAIN =========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[OK] Output directory:", OUTPUT_DIR)

    # 1) Load inputs
    zip3_pmf_df = load_zip3_pmf(PATHS["zip3_pmf"])
    fc_map_df = load_fc_assignment(PATHS["fc_assignment"])
    daily_units = load_daily_units(PATHS["daily_profile_csv"], DATES, PATHS["daily_profile_value_col"])

    # 2) Build ZIP3→daily, then FC→daily
    zip3_daily_df = split_to_zip3_daily(daily_units, zip3_pmf_df)
    fc_daily_df = aggregate_zip3_to_fc(zip3_daily_df, fc_map_df)
    if fc_daily_df.shape[1] == 0:
        raise RuntimeError("No FC columns produced — check ZIP3→FC assignment and PMF files")

    # 3) FC inventory (3-week @ 99%) — FUTURE window so Day 1 has a value
    fc_rows = []
    for fc in fc_daily_df.columns:
        series = fc_daily_df[fc].values
        for i, dt in enumerate(fc_daily_df.index):
            window = series[i:i+FC_AUTONOMY_DAYS]
            cyc, saf, tgt = inv_from_future_slice(window, cv=CV_DAILY, z=Z_BASELINE_FC)
            fc_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "fc_id": fc,
                "fc_cycle": cyc,
                "fc_safety": saf,
                "fc_target": tgt
            })
    fc_daily_targets_df = pd.DataFrame(fc_rows)

    # 4) Network inventory (8-week @ 99%) & DC complement
    arr_network = fc_daily_df.sum(axis=1).values  # equals daily_units.values
    net_rows = []
    for i, dt in enumerate(fc_daily_df.index):
        window = arr_network[i:i+NET_AUTONOMY_DAYS]
        net_c, net_s, net_t = inv_from_future_slice(window, cv=CV_DAILY, z=Z_BASELINE_NET)
        date_str = dt.strftime("%Y-%m-%d")
        fc_todays_total = float(fc_daily_targets_df.loc[fc_daily_targets_df["date"] == date_str, "fc_target"].sum())
        dc_inv = max(0.0, net_t - fc_todays_total)
        net_rows.append({
            "date": date_str,
            "network_cycle": net_c,
            "network_safety": net_s,
            "network_target": net_t,
            "dc_inventory": dc_inv
        })
    network_daily_df = pd.DataFrame(net_rows)

    # 5) Per-FC maxima + mean
    fc_max_df = (fc_daily_targets_df
                 .groupby("fc_id", as_index=False)["fc_target"]
                 .agg(fc_target_max="max", fc_target_mean="mean"))

    network_max_df = pd.DataFrame([{
        "network_target_max": float(network_daily_df["network_target"].max()),
        "network_target_mean": float(network_daily_df["network_target"].mean()),
        "dc_inventory_max": float(network_daily_df["dc_inventory"].max()),
        "dc_inventory_mean": float(network_daily_df["dc_inventory"].mean()),
    }])

    # 6) 7c — DAILY scenarios file (weeks ∈ {6,12}, service ∈ {50,68,95}), FCs keep baseline
    scen_rows = []
    for weeks in SCENARIO_WEEKS:
        Nd = weeks * 7
        for svc_label, z in SCENARIO_Z.items():
            for i, dt in enumerate(fc_daily_df.index):
                window = arr_network[i:i+Nd]
                if window.size == 0:
                    net_c = net_s = net_t = 0.0
                else:
                    net_c, net_s, net_t = inv_from_future_slice(window, cv=CV_DAILY, z=z)
                date_str = dt.strftime("%Y-%m-%d")
                fc_todays_total = float(fc_daily_targets_df.loc[fc_daily_targets_df["date"] == date_str, "fc_target"].sum())
                dc_inv = max(0.0, net_t - fc_todays_total)
                scen_rows.append({
                    "date": date_str,
                    "weeks": weeks,
                    "days": Nd,
                    "service_level": svc_label,
                    "z": z,
                    "network_cycle": net_c,
                    "network_safety": net_s,
                    "network_target": net_t,
                    "fc_total_target_today": fc_todays_total,
                    "dc_inventory": dc_inv
                })
    scenarios_daily_df = pd.DataFrame(scen_rows)

    # 7) Save all outputs
    fc_daily_targets_df.to_csv(os.path.join(OUTPUT_DIR, "task7_fc_daily_targets.csv"), index=False)
    network_daily_df.to_csv(os.path.join(OUTPUT_DIR, "task7_network_daily.csv"), index=False)
    fc_max_df.to_csv(os.path.join(OUTPUT_DIR, "task7_fc_maximums.csv"), index=False)
    network_max_df.to_csv(os.path.join(OUTPUT_DIR, "task7_network_maximums.csv"), index=False)
    scenarios_daily_df.to_csv(os.path.join(OUTPUT_DIR, "task7_scenarios_daily.csv"), index=False)

    print("\n✅ Saved to:", OUTPUT_DIR)
    print(" - task7_fc_daily_targets.csv")
    print(" - task7_network_daily.csv")
    print(" - task7_fc_maximums.csv  (now includes fc_target_mean)")
    print(" - task7_network_maximums.csv")
    print(" - task7_scenarios_daily.csv")

if __name__ == "__main__":
    main()
