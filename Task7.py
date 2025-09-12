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


# Load the PMF file
zip3_pmf = pd.read_csv(paths["zip3_pmf"])
print("Loaded zip3_pmf.csv:")
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

# --- Alternative scenarios ---
scenarios = [(6, 0.50), (6, 0.95), (12, 0.99)]
for weeks, prob in scenarios:
    N_days = weeks * 7
    z = ROBUSTNESS[prob]
    net_inv = compute_autonomy_inventory(np.sum(list(demands_fc.values()), axis=0), N_days, z)
    print(f"{weeks}-week at {int(prob*100)}% → {ceil(net_inv)} units")
