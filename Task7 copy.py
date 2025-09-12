import numpy as np
import pandas as pd
from math import ceil

# PARAMETERS
ROBUSTNESS = {0.50: 0.0, 0.68: 1.0, 0.95: 1.65, 0.99: 2.33}  # z levels
FC_AUTONOMY_DAYS = 21    # 3 weeks
NETWORK_AUTONOMY_DAYS = 56  # 8 weeks

paths = {
    "zip3_pmf": r"D:\ISyE6202_CW1\CW1\data\zip3_pmf.csv"
}

zip3_pmf = pd.read_csv(paths["zip3_pmf"])
print(zip3_pmf.head())

# --- Inventory Computation Functions ---
def compute_autonomy_inventory(daily_demand, N_days, z):
    """
    Compute target inventory given daily demand series, autonomy period (N_days), and z-value.
    """
    d_mean = np.mean(daily_demand)
    d_std = np.std(daily_demand)
    
    cycle_stock = d_mean * N_days
    safety_stock = z * d_std * np.sqrt(N_days)
    
    return cycle_stock + safety_stock


def compute_fc_inventories(demands_fc, fc_days=FC_AUTONOMY_DAYS, z=ROBUSTNESS[0.99]):
    """
    Compute autonomy-based inventory per FC.
    demands_fc: dict {FC_id: daily demand series (numpy array)}
    """
    fc_inventories = {}
    for fc, d_series in demands_fc.items():
        fc_inventories[fc] = compute_autonomy_inventory(d_series, fc_days, z)
    return fc_inventories


def compute_network_inventory(demands_fc, network_days=NETWORK_AUTONOMY_DAYS, z=ROBUSTNESS[0.99]):
    """
    Compute network-wide inventory and DC complement.
    """
    # Aggregate demand across FCs
    daily_network = np.sum(list(demands_fc.values()), axis=0)
    
    network_inventory = compute_autonomy_inventory(daily_network, network_days, z)
    fc_inventories = compute_fc_inventories(demands_fc, FC_AUTONOMY_DAYS, z)
    dc_inventory = max(0, network_inventory - sum(fc_inventories.values()))
    
    return {
        "network_total": network_inventory,
        "fc_inventories": fc_inventories,
        "dc_inventory": dc_inventory
    }


# --- Example Usage ---
# ⚠️ Placeholder: Simulated demand data for testing
# Replace this with real demand per FC once you aggregate from zip3_pmf.csv
np.random.seed(42)
demands_fc = {
    "FC-GA-303": np.random.normal(100, 15, 365),  # mean 100, std 15
    "FC-NY-134": np.random.normal(80, 10, 365),   # mean 80, std 10
    "FC-TX-799": np.random.normal(90, 12, 365)    # mean 90, std 12
}

# Compute default (3-week 99% FC + 8-week 99% network)
results = compute_network_inventory(demands_fc)

print("\n--- Results ---")
print("Network target inventory:", ceil(results["network_total"]))
print("FC inventories:", {fc: ceil(inv) for fc, inv in results["fc_inventories"].items()})
print("DC inventory:", ceil(results["dc_inventory"]))

# --- Save Task 7 results as CSV ---
fc_inv_df = pd.DataFrame([
    {"FC_ID": fc, "Inventory": ceil(inv)}
    for fc, inv in results["fc_inventories"].items()
])

extra_rows = pd.DataFrame([
    {"FC_ID": "DC", "Inventory": ceil(results["dc_inventory"])},
    {"FC_ID": "Network_Total", "Inventory": ceil(results["network_total"])}
])

output_df = pd.concat([fc_inv_df, extra_rows], ignore_index=True)
output_df.to_csv("task7_inventory_results.csv", index=False)
print("\nSaved FC/DC/Network results → task7_inventory_results.csv")

# --- Alternative scenarios ---
scenarios = []
for weeks, prob in [(6, 0.50), (6, 0.95), (12, 0.99)]:
    N_days = weeks * 7
    z = ROBUSTNESS[prob]
    net_inv = compute_autonomy_inventory(np.sum(list(demands_fc.values()), axis=0), N_days, z)
    scenarios.append({
        "Weeks": weeks,
        "Confidence": prob,
        "Inventory": ceil(net_inv)
    })
    print(f"{weeks}-week at {int(prob*100)}% → {ceil(net_inv)} units")

scenarios_df = pd.DataFrame(scenarios)
scenarios_df.to_csv("task7_scenarios.csv", index=False)
print("Saved scenario analysis → task7_scenarios.csv")
