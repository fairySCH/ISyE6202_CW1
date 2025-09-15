"""
Dobda Case — Task 3 (a, b, c) Solution Script
Author: ChatGPT
Inputs (expected in CW1/data):
  - zip3_pmf.csv
  - zip3_market.csv  (contains Market column: Primary / Secondary / Tertiary)
  - zip3_coordinates.csv
  - fc_zip3_distance.csv

Outputs (written to CW1/output/task3/):
  - (a) cluster plots per network: task3a_{network}.png
  - (b) tables:
        task3b_fc_demand_share_{network}.csv
        task3b_fc_market_demand_share_{network}.csv
  - (c) tables:
        task3c_market_distance_distribution_{network}.csv
        task3c_fc_market_distance_distribution_{network}.csv
  - helper: assignment_{network}.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]   # project root
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output" / "task3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORKS = {
    "1FC": {"GA-303": 303},
    "4FC": {"GA-303": 303, "NY-134": 134, "TX-799": 799, "UT-841": 841},
    "15FC": {
        "AZ-852": 852, "CA-900": 900, "CA-945": 945, "CO-802": 802,
        "FL-331": 331, "GA-303": 303, "IL-606": 606, "MA-021": 21,
        "MI-481": 481, "NC-275": 275, "NJ-070": 70,
        "TX-750": 750, "TX-770": 770, "UT-841": 841, "WA-980": 980,
    },
}

BUCKET_EDGES = [-np.inf, 50, 150, 300, 600, 1000, 1400, 1800, np.inf]
BUCKET_LABELS = ["<50","51-150","151-300","301-600",
                 "601-1000","1001-1400","1401-1800",">1800"]

# -----------------------------
# Load data
# -----------------------------
zip3_pmf = pd.read_csv(DATA_DIR / "zip3_pmf.csv")
zip3_market = pd.read_csv(DATA_DIR / "zip3_market.csv")
zip3_coords = pd.read_csv(DATA_DIR / "zip3_coordinates.csv")
dist = pd.read_csv(DATA_DIR / "fc_zip3_distance.csv")

# Normalize column names
zip3_pmf.columns = [c.strip().lower() for c in zip3_pmf.columns]
zip3_market.columns = [c.strip().lower() for c in zip3_market.columns]
zip3_coords.columns = [c.strip().lower() for c in zip3_coords.columns]
dist.columns = [c.strip().lower() for c in dist.columns]

# Ensure correct mapping for demand PMF
if "pmf" not in zip3_pmf.columns:
    zip3_pmf = zip3_pmf.rename(columns={zip3_pmf.columns[-1]: "pmf"})

# ✅ Explicitly use the "market" column for Primary/Secondary/Tertiary
if "market" not in zip3_market.columns:
    raise ValueError("zip3_market.csv must contain a 'Market' column with values Primary/Secondary/Tertiary")
zip3_market = zip3_market.rename(columns={"market": "market_type"})
zip3_market["market_type"] = zip3_market["market_type"].astype(str).str.strip().str.title()

valid_markets = {"Primary","Secondary","Tertiary"}
bad_values = set(zip3_market["market_type"].unique()) - valid_markets
if bad_values:
    raise ValueError(f"Unexpected market_type values: {bad_values}")

# Normalize coordinates
if "lat" not in zip3_coords.columns or "lon" not in zip3_coords.columns:
    rename_map = {}
    if "latitude" in zip3_coords.columns: rename_map["latitude"] = "lat"
    if "longitude" in zip3_coords.columns: rename_map["longitude"] = "lon"
    if "x" in zip3_coords.columns: rename_map["x"] = "lon"
    if "y" in zip3_coords.columns: rename_map["y"] = "lat"
    zip3_coords = zip3_coords.rename(columns=rename_map)

# Fix distance table if wide format
if "fc_zip3" not in dist.columns:
    id_col = [c for c in dist.columns if "zip" in c][0]
    value_cols = [c for c in dist.columns if c != id_col]
    dist = dist.melt(id_vars=[id_col], value_vars=value_cols,
                     var_name="fc_zip3", value_name="distance_miles")
    dist = dist.rename(columns={id_col: "zip3"})
    dist["fc_zip3"] = dist["fc_zip3"].astype(str).str.extract(r'(\d+)').astype(int)

# Type coercion
zip3_pmf["zip3"] = zip3_pmf["zip3"].astype(int)
zip3_market["zip3"] = zip3_market["zip3"].astype(int)
zip3_coords["zip3"] = zip3_coords["zip3"].astype(int)
dist["zip3"] = dist["zip3"].astype(int)
dist["fc_zip3"] = dist["fc_zip3"].astype(int)
dist["distance_miles"] = pd.to_numeric(dist["distance_miles"], errors="coerce")

# Merge base data
base = zip3_pmf.merge(zip3_market, on="zip3", how="inner") \
               .merge(zip3_coords, on="zip3", how="left")
base["pmf_norm"] = base["pmf"] / base["pmf"].sum()

# -----------------------------
# Functions
# -----------------------------
def assign_preferred_fc(network_dict):
    """Assign each ZIP3 to the closest FC and compute distance bucket."""
    fc_zip3s = list(network_dict.values())
    dsub = dist[dist["fc_zip3"].isin(fc_zip3s)].copy()
    wide = dsub.pivot_table(index="zip3", columns="fc_zip3",
                            values="distance_miles", aggfunc="min")
    nearest_fc_zip3 = wide.idxmin(axis=1)
    nearest_dist = wide.min(axis=1)
    inv_map = {zip3: name for name, zip3 in network_dict.items()}
    assign = base.merge(nearest_fc_zip3.rename("preferred_fc_zip3"), on="zip3", how="left") \
                 .merge(nearest_dist.rename("preferred_fc_distance"), on="zip3", how="left")
    assign["preferred_fc"] = assign["preferred_fc_zip3"].map(inv_map)
    assign["distance_bucket"] = pd.cut(
        assign["preferred_fc_distance"],
        bins=BUCKET_EDGES,
        labels=BUCKET_LABELS,
        right=True,
        include_lowest=True
    )
    assign["distance_bucket"] = pd.Categorical(assign["distance_bucket"],
                                               categories=BUCKET_LABELS, ordered=True)
    return assign

def plot_clusters(assign, network_name):
    """Plot ZIP clusters by preferred FC with FCs shown as triangles."""
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())

    # Map features
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)

    # Color palette
    fc_names = sorted(assign["preferred_fc"].dropna().unique())
    palette = sns.color_palette("husl", n_colors=len(fc_names))
    color_map = {fc: palette[i] for i, fc in enumerate(fc_names)}

    # ZIP dots
    for fc_name, grp in assign.groupby("preferred_fc"):
        ax.scatter(grp["lon"], grp["lat"], s=6, alpha=0.4,
                   color=color_map.get(fc_name, "gray"),
                   transform=ccrs.PlateCarree(), label=fc_name)

    # FC triangles
    fc_map = assign.dropna(subset=["preferred_fc_zip3"]) \
                   .groupby("preferred_fc")["preferred_fc_zip3"].first().to_dict()
    for fc_name, fc_zip in fc_map.items():
        row = base[base["zip3"] == int(fc_zip)]
        if len(row) > 0:
            ax.scatter(float(row["lon"].iloc[0]), float(row["lat"].iloc[0]),
                       s=200, marker="^", color=color_map.get(fc_name, "red"),
                       edgecolor="black", linewidths=0.7,
                       label=f"{fc_name} (FC)",
                       transform=ccrs.PlateCarree(), zorder=5)

    # Legend
    ax.legend(markerscale=0.7, fontsize=7, ncol=1, frameon=False,
              loc="center right", bbox_to_anchor=(-0.1, 0.5))
    plt.subplots_adjust(left=0.15)

    ax.set_title(f"Task 3a — ZIP3 Clusters by Preferred FC ({network_name})", fontsize=12)
    plt.savefig(OUT_DIR / f"task3a_{network_name}_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def compute_task3b(assign, network_name):
    """Compute aggregated demand share per FC and per FC×Market combination."""
    # FC-level
    g_fc = assign.groupby("preferred_fc", dropna=False)["pmf_norm"].sum().reset_index()
    g_fc = g_fc.rename(columns={"pmf_norm": "demand_share"})
    g_fc.to_csv(OUT_DIR / f"task3b_fc_demand_share_{network_name}.csv", index=False)

    # FC × Market-level
    assign = assign.copy()
    cats = ["Primary","Secondary","Tertiary"]
    assign["market_type"] = pd.Categorical(assign["market_type"], categories=cats, ordered=True)
    g_fc_mt = assign.groupby(["preferred_fc","market_type"], dropna=False)["pmf_norm"].sum().reset_index()
    g_fc_mt = g_fc_mt.rename(columns={"pmf_norm": "demand_share"}).sort_values(["preferred_fc","market_type"])
    g_fc_mt.to_csv(OUT_DIR / f"task3b_fc_market_demand_share_{network_name}.csv", index=False)

def compute_task3c(assign, network_name):
    """Compute demand distribution by market type and distance bucket."""
    sub = assign.dropna(subset=["distance_bucket"]).copy()
    sub["distance_bucket"] = pd.Categorical(sub["distance_bucket"],
                                            categories=BUCKET_LABELS, ordered=True)

    # Market × Distance
    mt = sub.groupby(["market_type","distance_bucket"], dropna=False)["pmf_norm"].sum()
    all_mt = sub["market_type"].dropna().unique().tolist()
    full_idx = pd.MultiIndex.from_product([all_mt, BUCKET_LABELS],
                                          names=["market_type","distance_bucket"])
    mt = mt.reindex(full_idx, fill_value=0).reset_index()
    mt = mt.rename(columns={"pmf_norm": "demand_share"})
    mt.to_csv(OUT_DIR / f"task3c_market_distance_distribution_{network_name}.csv", index=False)

    # FC × Market × Distance
    fcm = sub.groupby(["preferred_fc","market_type","distance_bucket"], dropna=False)["pmf_norm"].sum()
    all_fc = sub["preferred_fc"].dropna().unique().tolist()
    full_idx2 = pd.MultiIndex.from_product([all_fc, all_mt, BUCKET_LABELS],
                                           names=["preferred_fc","market_type","distance_bucket"])
    fcm = fcm.reindex(full_idx2, fill_value=0).reset_index()
    fcm = fcm.rename(columns={"pmf_norm": "demand_share"})
    fcm.to_csv(OUT_DIR / f"task3c_fc_market_distance_distribution_{network_name}.csv", index=False)

def run_for_network(network_name, fc_dict):
    assign = assign_preferred_fc(fc_dict)
    assign.to_csv(OUT_DIR / f"assignment_{network_name}.csv", index=False)
    plot_clusters(assign, network_name)
    compute_task3b(assign, network_name)
    compute_task3c(assign, network_name)

def main():
    for net, fcs in NETWORKS.items():
        run_for_network(net, fcs)

if __name__ == "__main__":
    main()
