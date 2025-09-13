# =========================
# Task 8 — Assembly Factory (AF) capacity vs smoothing, Lead Time = 0
# =========================
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------
# 0) Configuration
# -------------------------
project_root_folder = Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1")

# Point directly to your Task-7 daily file (must have: date, fc_id, fc_cycle, fc_target)
input_task7_csv = Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7/task7_fc_daily_targets.csv")

# Task 8 outputs folder
output_folder_task8 = project_root_folder / "output" / "task8"
output_folder_task8.mkdir(parents=True, exist_ok=True)

# Inventory starting condition at the Distribution Center (units)
initial_dc_on_hand_inventory = 0.0

# Choose network aggregation mode
aggregate_network = False
fulfillment_center_filter = None  # e.g., "AZ-852" or None

# Column names present in the Task-7 file
column_name_date = "date"
column_name_fulfillment_center_id = "fc_id"
column_name_robust_daily_demand_task7 = "fc_cycle"    # robust daily demand (99th percentile)
column_name_autonomy_target_stock_task7 = "fc_target"  # target autonomy stock (8-week-99%)

# -------------------------
# 1) Load Task-7 daily file
# -------------------------
if not input_task7_csv.exists():
    raise FileNotFoundError(f"Task-7 CSV not found at: {input_task7_csv}")

task7_dataframe_raw = pd.read_csv(input_task7_csv)
required_cols = [column_name_date, column_name_robust_daily_demand_task7, column_name_autonomy_target_stock_task7]
missing = [c for c in required_cols if c not in task7_dataframe_raw.columns]
if missing:
    raise RuntimeError(f"Input file is missing required columns: {missing}")

task7_dataframe_raw[column_name_date] = pd.to_datetime(task7_dataframe_raw[column_name_date])

# -------------------------
# 2) Prepare daily series for Task 8
# -------------------------
if aggregate_network:
    # Sum across all fulfillment centers per date
    daily_dataframe = (
        task7_dataframe_raw
        .groupby(column_name_date, as_index=False)[[column_name_robust_daily_demand_task7,
                                                    column_name_autonomy_target_stock_task7]]
        .sum()
        .sort_values(column_name_date)
        .reset_index(drop=True)
    )
    daily_dataframe = daily_dataframe.rename(columns={
        column_name_robust_daily_demand_task7: "robust_daily_demand",
        column_name_autonomy_target_stock_task7: "autonomy_target_stock_8week_99pct"
    })
else:
    # Single fulfillment center pathway
    if fulfillment_center_filter is not None and column_name_fulfillment_center_id in task7_dataframe_raw.columns:
        subset = task7_dataframe_raw[task7_dataframe_raw[column_name_fulfillment_center_id] == fulfillment_center_filter].copy()
        if subset.empty:
            raise RuntimeError(f"No rows found for fulfillment_center_id='{fulfillment_center_filter}'.")
        daily_dataframe = subset.sort_values(column_name_date).reset_index(drop=True)
        print(f"[INFO] Using fulfillment_center_id: {fulfillment_center_filter}")
    else:
        if column_name_fulfillment_center_id in task7_dataframe_raw.columns:
            first_fc_id = task7_dataframe_raw[column_name_fulfillment_center_id].iloc[0]
            daily_dataframe = (
                task7_dataframe_raw[task7_dataframe_raw[column_name_fulfillment_center_id] == first_fc_id]
                .sort_values(column_name_date)
                .reset_index(drop=True)
            )
            print(f"[INFO] No filter set; using first fulfillment_center_id found: {first_fc_id}")
        else:
            # Already single-DC data
            daily_dataframe = task7_dataframe_raw.sort_values(column_name_date).reset_index(drop=True)

    daily_dataframe = daily_dataframe.rename(columns={
        column_name_robust_daily_demand_task7: "robust_daily_demand",
        column_name_autonomy_target_stock_task7: "autonomy_target_stock_8week_99pct"
    })[[column_name_date, "robust_daily_demand", "autonomy_target_stock_8week_99pct"]]

# Clean and standardize
daily_dataframe = daily_dataframe.rename(columns={column_name_date: "date"})
daily_dataframe["robust_daily_demand"] = pd.to_numeric(
    daily_dataframe["robust_daily_demand"], errors="coerce"
).fillna(0.0).clip(lower=0.0)

daily_dataframe["autonomy_target_stock_8week_99pct"] = pd.to_numeric(
    daily_dataframe["autonomy_target_stock_8week_99pct"], errors="coerce"
).fillna(0.0).clip(lower=0.0)

daily_dataframe = daily_dataframe.sort_values("date").reset_index(drop=True)
if len(daily_dataframe) == 0:
    raise RuntimeError("No daily rows to process after filtering/aggregation.")

# -------------------------
# 3) Strategy A — Chase the autonomy target (Lead Time = 0)
# -------------------------
robust_daily_demand_array = daily_dataframe["robust_daily_demand"].to_numpy(dtype=float)
autonomy_target_stock_array = daily_dataframe["autonomy_target_stock_8week_99pct"].to_numpy(dtype=float)
time_horizon_days = len(daily_dataframe)

dc_on_hand_inventory_start_of_day = np.zeros(time_horizon_days + 1, dtype=float)
dc_on_hand_inventory_start_of_day[0] = float(initial_dc_on_hand_inventory)

af_daily_production_chase = np.zeros(time_horizon_days, dtype=float)
dc_inventory_after_arrival_before_demand_chase = np.zeros(time_horizon_days, dtype=float)

for day_index in range(time_horizon_days):
    # Minimal production today so that (inventory after arrivals) meets today's target
    af_daily_production_chase[day_index] = max(
        0.0,
        autonomy_target_stock_array[day_index] - dc_on_hand_inventory_start_of_day[day_index]
    )
    # Inventory at DC after arrival, before serving demand
    dc_inventory_after_arrival_before_demand_chase[day_index] = (
        dc_on_hand_inventory_start_of_day[day_index] + af_daily_production_chase[day_index]
    )
    # End-of-day on-hand inventory after serving today's demand
    dc_on_hand_inventory_start_of_day[day_index + 1] = max(
        0.0,
        dc_inventory_after_arrival_before_demand_chase[day_index] - robust_daily_demand_array[day_index]
    )

peak_af_daily_capacity_chase = float(af_daily_production_chase.max())
peak_dc_storage_within_day_chase = float(dc_inventory_after_arrival_before_demand_chase.max())
peak_dc_storage_end_of_day_chase = float(dc_on_hand_inventory_start_of_day[1:].max())

# -------------------------
# 4) Strategy B — Leveled (steady) production (Lead Time = 0)
# -------------------------
# Minimal feasible constant rate:
#   constant_rate_star = max_t (autonomy_target_stock_8week_99pct_t - I0 + sum_{k=0}^{t-1} robust_daily_demand_k) / (t+1)
cumulative_demand_previous_days = np.concatenate(
    ([0.0], np.cumsum(robust_daily_demand_array)[:-1])
)
constant_rate_candidates = (
    (autonomy_target_stock_array - float(initial_dc_on_hand_inventory) + cumulative_demand_previous_days)
    / (np.arange(time_horizon_days) + 1)
)
constant_rate_star = float(max(0.0, np.max(constant_rate_candidates)))

dc_on_hand_inventory_start_of_day_level = np.zeros(time_horizon_days + 1, dtype=float)
dc_on_hand_inventory_start_of_day_level[0] = float(initial_dc_on_hand_inventory)

dc_inventory_after_arrival_before_demand_level = np.zeros(time_horizon_days, dtype=float)

for day_index in range(time_horizon_days):
    # After-arrival (since lead time = 0) inventory with constant production
    dc_inventory_after_arrival_before_demand_level[day_index] = (
        dc_on_hand_inventory_start_of_day_level[day_index] + constant_rate_star
    )
    # End-of-day inventory after serving demand
    dc_on_hand_inventory_start_of_day_level[day_index + 1] = max(
        0.0,
        dc_inventory_after_arrival_before_demand_level[day_index] - robust_daily_demand_array[day_index]
    )

peak_af_daily_capacity_level = constant_rate_star
peak_dc_storage_within_day_level = float(dc_inventory_after_arrival_before_demand_level.max())
peak_dc_storage_end_of_day_level = float(dc_on_hand_inventory_start_of_day_level[1:].max())

# -------------------------
# 5) Build outputs (UPPERCASE column names)
# -------------------------
chase_output_dataframe = pd.DataFrame({
    "DATE": daily_dataframe["date"],
    "ROBUST_DAILY_DEMAND": robust_daily_demand_array,
    "AUTONOMY_TARGET_STOCK_8WEEK_99PCT": autonomy_target_stock_array,
    "AF_DAILY_PRODUCTION_CHASE": af_daily_production_chase,
    "DC_INVENTORY_AFTER_ARRIVAL_BEFORE_DEMAND_CHASE": dc_inventory_after_arrival_before_demand_chase,
    "DC_ON_HAND_INVENTORY_END_OF_DAY_CHASE": dc_on_hand_inventory_start_of_day[1:],
})

level_output_dataframe = pd.DataFrame({
    "DATE": daily_dataframe["date"],
    "ROBUST_DAILY_DEMAND": robust_daily_demand_array,
    "AUTONOMY_TARGET_STOCK_8WEEK_99PCT": autonomy_target_stock_array,
    "AF_DAILY_PRODUCTION_LEVELED_CONSTANT_RATE": constant_rate_star,
    "DC_INVENTORY_AFTER_ARRIVAL_BEFORE_DEMAND_LEVELED": dc_inventory_after_arrival_before_demand_level,
    "DC_ON_HAND_INVENTORY_END_OF_DAY_LEVELED": dc_on_hand_inventory_start_of_day_level[1:],
})

summary_output_dataframe = pd.DataFrame({
    "METRIC_NAME": [
        "AF_PEAK_DAILY_CAPACITY",
        "DC_PEAK_STORAGE_WITHIN_DAY",
        "DC_PEAK_STORAGE_END_OF_DAY"
    ],
    "STRATEGY_A_CHASE_VALUE": [
        peak_af_daily_capacity_chase,
        peak_dc_storage_within_day_chase,
        peak_dc_storage_end_of_day_chase
    ],
    "STRATEGY_B_LEVELED_VALUE": [
        peak_af_daily_capacity_level,
        peak_dc_storage_within_day_level,
        peak_dc_storage_end_of_day_level
    ]
})

# -------------------------
# 6) Save files
# -------------------------
chase_output_path = output_folder_task8 / "task8_AF_Strategy_A.csv"
level_output_path = output_folder_task8 / "task8_AF_Strategy_B.csv"
summary_output_path = output_folder_task8 / "task8_AF_Summary_L0.csv"

chase_output_dataframe.to_csv(chase_output_path, index=False)
level_output_dataframe.to_csv(level_output_path, index=False)
summary_output_dataframe.to_csv(summary_output_path, index=False)

print("== Task 8 (Lead Time = 0) — Results ==")
print(summary_output_dataframe.to_string(index=False))
print(f"\nFiles written to: {output_folder_task8}")
