# task4.py
# Dobda Case — Task 4 (a,b,c,d) Automation

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try Cartopy (map background). If not available, fallback to plain scatter.
_HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    _HAS_CARTOPY = False

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # project/CW1/
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "task4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORKS = {
    "1FC": {"GA-303": 303},
    "4FC": {"GA-303": 303, "NY-134": 134, "TX-799": 799, "UT-841": 841},
    "15FC": {
        "AZ-852": 852, "CA-900": 900, "CA-945": 945, "CO-802": 802,
        "FL-331": 331, "GA-303": 303, "IL-606": 606, "MA-021": 21,
        "MI-481": 481, "NC-275": 275, "NJ-070": 70,
        "TX-750": 750, "TX-770": 770, "UT-841": 841, "WA-980": 980
    }
}

BUCKET_EDGES = [-np.inf, 50, 150, 300, 600, 1000, 1400, 1800, np.inf]
BUCKET_LABELS = ["<50","51-150","151-300","301-600","601-1000","1001-1400","1401-1800",">1800"]
BUCKET_ORD = {lab:i for i,lab in enumerate(BUCKET_LABELS)}

def bucketize_series(dist_series):
    return pd.cut(dist_series, bins=BUCKET_EDGES, labels=BUCKET_LABELS,
                  right=True, include_lowest=True)

# -----------------------------
# Load data
# -----------------------------
def _norm_cols(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

zip3_pmf = _norm_cols(pd.read_csv(DATA_DIR / "zip3_pmf.csv"))
zip3_market = _norm_cols(pd.read_csv(DATA_DIR / "zip3_market.csv"))
zip3_coords = _norm_cols(pd.read_csv(DATA_DIR / "zip3_coordinates.csv"))
dist = _norm_cols(pd.read_csv(DATA_DIR / "fc_zip3_distance.csv"))

# Normalize columns
if "pmf" not in zip3_pmf.columns:
    zip3_pmf = zip3_pmf.rename(columns={zip3_pmf.columns[-1]: "pmf"})
if "market_type" not in zip3_market.columns:
    zip3_market = zip3_market.rename(columns={zip3_market.columns[-1]: "market_type"})
if "lat" not in zip3_coords.columns or "lon" not in zip3_coords.columns:
    rename_map = {}
    if "latitude" in zip3_coords.columns: rename_map["latitude"] = "lat"
    if "longitude" in zip3_coords.columns: rename_map["longitude"] = "lon"
    if "x" in zip3_coords.columns: rename_map["x"] = "lon"
    if "y" in zip3_coords.columns: rename_map["y"] = "lat"
    zip3_coords = zip3_coords.rename(columns=rename_map)

# Distance tidy (long format)
if "distance_miles" not in dist.columns:
    id_col = [c for c in dist.columns if "zip" in c][0]
    value_cols = [c for c in dist.columns if c != id_col]
    dist = dist.melt(id_vars=[id_col], value_vars=value_cols,
                     var_name="fc_zip3", value_name="distance_miles")
    dist = dist.rename(columns={id_col: "zip3"})
    dist["fc_zip3"] = dist["fc_zip3"].astype(str).str.extract(r'(\d+)').astype(int)

# Types
zip3_pmf["zip3"] = zip3_pmf["zip3"].astype(int)
zip3_market["zip3"] = zip3_market["zip3"].astype(int)
zip3_coords["zip3"] = zip3_coords["zip3"].astype(int)
dist["zip3"] = dist["zip3"].astype(int)
dist["fc_zip3"] = dist["fc_zip3"].astype(int)
dist["distance_miles"] = pd.to_numeric(dist["distance_miles"], errors="coerce")

# Base frame
base = zip3_pmf.merge(zip3_market, on="zip3").merge(zip3_coords, on="zip3", how="left")
base["pmf_norm"] = base["pmf"] / base["pmf"].sum()

# -----------------------------
# Preferred assignment (Task 3 base)
# -----------------------------
def compute_preferred_assignment(network_dict):
    fc_zip3s = list(network_dict.values())
    dsub = dist[dist["fc_zip3"].isin(fc_zip3s)]
    wide = dsub.pivot_table(index="zip3", columns="fc_zip3", values="distance_miles", aggfunc="min")
    nearest_fc_zip3 = wide.idxmin(axis=1)
    nearest_dist = wide.min(axis=1)
    inv_map = {zip3: name for name, zip3 in network_dict.items()}
    assign = base.merge(nearest_fc_zip3.rename("preferred_fc_zip3"), on="zip3", how="left") \
                 .merge(nearest_dist.rename("preferred_fc_distance"), on="zip3", how="left")
    assign["preferred_fc"] = assign["preferred_fc_zip3"].map(inv_map)
    assign["preferred_bucket"] = bucketize_series(assign["preferred_fc_distance"])
    return assign

# -----------------------------
# Task 4 core, plots, metrics
# -----------------------------
# (여기서는 run_for_network 내부에서 print 찍어 저장 확인)

def run_for_network(network_name, network_dict):
    print(f"▶ Running network {network_name} ...")

    assign = compute_preferred_assignment(network_dict)
    out_file = OUT_DIR / f"task4_assignment_{network_name}.csv"
    assign.to_csv(out_file, index=False)
    print(f"  Saved assignment -> {out_file}")

    # 예시: cluster plot
    fig, ax = plt.subplots()
    ax.scatter(assign["lon"], assign["lat"], s=5)
    ax.set_title(f"Task 4a Example — {network_name}")
    out_path = OUT_DIR / f"task4a_{network_name}_clusters_map.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot -> {out_path}")

def main():
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", OUT_DIR)
    for net, fcs in NETWORKS.items():
        run_for_network(net, fcs)

if __name__ == "__main__":
    main()
