# task7_full_pipeline_fixed.py
import os
import numpy as np
import pandas as pd
from math import ceil

# ---------------------------
# PARAMETERS (adjustable)
# ---------------------------
# Z-values (service levels)
ROBUSTNESS = {0.50: 0.0, 0.68: 0.468, 0.95: 1.645, 0.99: 2.33}

# Autonomy days
FC_AUTONOMY_DAYS = 21         # 3 weeks
NETWORK_AUTONOMY_DAYS = 56    # 8 weeks

# Coefficient of variation assumption for per-day demand uncertainty
CV_DAILY = 0.15

# Planning horizon
YEAR = 2026
DATES = pd.date_range(start=f"{YEAR}-01-01", end=f"{YEAR}-12-31", freq='D')

# File paths (adjust to your local layout)
paths = {
    "zip3_pmf": r"D:\ISyE6202_CW1\CW1\data\zip3_pmf.csv",
    "fc_zip3_assignment": r"D:\ISyE6202_CW1\CW1\data\dobda_task3_outputs\task3c_fc_market_distance_distribution_15FC.csv",  # must exist
    "demand_seasonalities": r"D:\ISyE6202_CW1\CW1\data\dobda_demand_seasonalities.csv"  # optional
}

# Market parameters (median scenario)
TOTAL_MARKET_TODAY = 2_000_000
CURRENT_MARKET_SHARE = 0.036
OVERALL_MARKET_GROW_MEDIAN = 0.075
DODBA_MARKET_SHARE_GROW_MEDIAN = 0.20


# ---------------------------
# Helper functions
# ---------------------------
def check_file(path, friendly_name=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}" +
                                (f" ({friendly_name})" if friendly_name else ""))
    return path

def normalize_headers(df):
    """Convert all column names to lowercase and strip spaces."""
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_zip3_pmf(path):
    path = check_file(path, "zip3 PMF")
    df = pd.read_csv(path, dtype=str)
    df = normalize_headers(df)
    if "zip3" not in df.columns or "pmf" not in df.columns:
        raise KeyError("zip3_pmf.csv must contain 'zip3' and 'pmf' columns (case-insensitive).")
    df["pmf"] = df["pmf"].astype(float)
    return df[["zip3", "pmf"]]

def load_fc_assignment(path):
    """Load ZIP3 → FC mapping. Accepts 'preferred_fc' instead of 'fc_id'."""
    path = check_file(path, "fc_zip3_assignment")
    df = pd.read_csv(path, dtype=str)
    df = normalize_headers(df)
    print("DEBUG: columns in fc_zip3_assignment =", df.columns.tolist())

    # If file has preferred_fc instead of fc_id → rename
    if "preferred_fc" in df.columns and "fc_id" not in df.columns:
        df = df.rename(columns={"preferred_fc": "fc_id"})

    if "zip3" not in df.columns or "fc_id" not in df.columns:
        raise KeyError("fc_zip3_assignment.csv must contain 'zip3' and 'fc_id' (or 'preferred_fc').")

    return df[["zip3", "fc_id"]]

def compute_annual_dobda_units():
    """Compute Dobda annual demand under median scenario."""
    overall_next = TOTAL_MARKET_TODAY * (1 + OVERALL_MARKET_GROW_MEDIAN)
    dodba_share_next = CURRENT_MARKET_SHARE * (1 + DODBA_MARKET_SHARE_GROW_MEDIAN)
    return overall_next * dodba_share_next

def allocate_yearly_to_zip3(zip3_pmf_df, annual_units):
    df = zip3_pmf_df.copy()
    df["yearly_demand"] = df["pmf"] * annual_units
    return df

def build_daily_factors(dates):
    """Flat daily factors (no seasonality)."""
    factors = np.ones(len(dates)) / len(dates)
    return pd.Series(factors, index=dates, name="day_factor")

def produce_zip3_daily_matrix(zip3_yearly_df, day_factors):
    mat = {}
    for _, row in zip3_yearly_df.iterrows():
        zip3 = row["zip3"]
        yearly = row["yearly_demand"]
        mat[zip3] = day_factors.values * yearly
    return pd.DataFrame(mat, index=day_factors.index)

def aggregate_zip3_to_fc(zip3_daily_df, fc_assignment_df):
    mapping = fc_assignment_df.set_index("zip3")["fc_id"].to_dict()
    missing = [z for z in zip3_daily_df.columns if z not in mapping]
    if missing:
        print(f"⚠️ Warning: {len(missing)} ZIP3 codes have no FC assignment (dropping them).")
        zip3_daily_df = zip3_daily_df.drop(columns=missing)
    fc_cols = {}
    for zip3 in zip3_daily_df.columns:
        fc = mapping.get(zip3)
        if fc:
            fc_cols.setdefault(fc, []).append(zip3)
    fc_daily = {fc: zip3_daily_df[zip3s].sum(axis=1).values for fc, zip3s in fc_cols.items()}
    return pd.DataFrame(fc_daily, index=zip3_daily_df.index)

def daily_target_from_future_slice(future_expected_array, cv=CV_DAILY, z=ROBUSTNESS[0.99]):
    future_expected = np.array(future_expected_array, dtype=float)
    cycle = future_expected.sum()
    var_total = np.sum((cv * future_expected) ** 2)
    safety = z * np.sqrt(var_total)
    return cycle, safety, cycle + safety


# ---------------------------
# Main pipeline
# ---------------------------
def main():
    # Load data
    zip3_pmf_df = load_zip3_pmf(paths["zip3_pmf"])
    fc_assign_df = load_fc_assignment(paths["fc_zip3_assignment"])

    # Compute annual Dobda demand
    annual_units = compute_annual_dobda_units()
    print(f"Annual Dobda demand (median scenario): {int(annual_units):,} units")

    # Build daily factors (flat)
    day_factors = build_daily_factors(DATES)

    # Allocate demand
    zip3_yearly = allocate_yearly_to_zip3(zip3_pmf_df, annual_units)
    zip3_daily_df = produce_zip3_daily_matrix(zip3_yearly, day_factors)
    fc_daily_df = aggregate_zip3_to_fc(zip3_daily_df, fc_assign_df)

    # Compute per-FC daily inventory targets
    fc_daily_targets = []
    for fc in fc_daily_df.columns:
        arr = fc_daily_df[fc].values
        for i, date in enumerate(fc_daily_df.index):
            future = arr[i:i+FC_AUTONOMY_DAYS]
            if future.size == 0:
                cycle = safety = total = 0.0
            else:
                cycle, safety, total = daily_target_from_future_slice(future, z=ROBUSTNESS[0.99])
            fc_daily_targets.append({
                "date": date.strftime("%Y-%m-%d"),
                "fc_id": fc,
                "fc_cycle": cycle,
                "fc_safety": safety,
                "fc_target": total
            })
    fc_daily_targets_df = pd.DataFrame(fc_daily_targets)

    # Compute network + DC inventory targets
    arr_network = fc_daily_df.sum(axis=1).values
    network_records = []
    for i, date in enumerate(fc_daily_df.index):
        future = arr_network[i:i+NETWORK_AUTONOMY_DAYS]
        if future.size == 0:
            net_cycle = net_safety = net_total = dc_inv = 0.0
        else:
            net_cycle, net_safety, net_total = daily_target_from_future_slice(future, z=ROBUSTNESS[0.99])
            fc_targets_same_day = fc_daily_targets_df.loc[fc_daily_targets_df["date"] == date.strftime("%Y-%m-%d"), "fc_target"].sum()
            dc_inv = max(0.0, net_total - fc_targets_same_day)
        network_records.append({
            "date": date.strftime("%Y-%m-%d"),
            "network_cycle": net_cycle,
            "network_safety": net_safety,
            "network_target": net_total,
            "dc_inventory": dc_inv
        })
    network_daily_df = pd.DataFrame(network_records)

    # Compute maxima
    fc_max = fc_daily_targets_df.groupby("fc_id")["fc_target"].max().reset_index(name="fc_target_max")
    network_max = {
        "network_target_max": network_daily_df["network_target"].max(),
        "dc_inventory_max": network_daily_df["dc_inventory"].max()
    }

    # Save results
    out_dir = os.getcwd()
    fc_daily_targets_df.to_csv(os.path.join(out_dir, "task7_fc_daily_targets.csv"), index=False)
    network_daily_df.to_csv(os.path.join(out_dir, "task7_network_daily.csv"), index=False)
    fc_max.to_csv(os.path.join(out_dir, "task7_fc_maximums.csv"), index=False)
    pd.DataFrame([network_max]).to_csv(os.path.join(out_dir, "task7_network_maximums.csv"), index=False)

    print("\n✅ Results saved:")
    print(" - task7_fc_daily_targets.csv")
    print(" - task7_network_daily.csv")
    print(" - task7_fc_maximums.csv")
    print(" - task7_network_maximums.csv")
    print("\nExample maxima:")
    print(fc_max.head())
    print(network_max)


if __name__ == "__main__":
    main()
