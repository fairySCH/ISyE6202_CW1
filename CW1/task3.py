
"""
Dobda Case — Task 3 (a, b, c) Solution Script
Author: ChatGPT
Inputs (expected in /mnt/data):
  - zip3_pmf.csv              # demand PMF for each 3-digit ZIP
  - zip3_market.csv           # market type for each 3-digit ZIP
  - zip3_coordinates.csv      # lat/lon of each 3-digit ZIP
  - fc_zip3_distance.csv      # distances (miles) from FC zip3 to each ZIP3

Outputs (written to /mnt/data/dobda_task3_outputs):
  - (a) cluster plots per network: task3a_{network}.png
  - (b) tables:
        task3b_fc_demand_share_{network}.csv
        task3b_fc_market_demand_share_{network}.csv
  - (c) tables:
        task3c_market_distance_distribution_{network}.csv
        task3c_fc_market_distance_distribution_{network}.csv   (optional granular)
  - (helper) assignments and merged data for auditing:
        assignment_{network}.csv  (ZIP3 -> preferred FC, distance, bucket, market, pmf)
Usage:
  python task3_solution.py  # will run all networks and export outputs
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path("./data")
OUT_DIR = BASE_DIR / "dobda_task3_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# FC network definitions (FC names -> zip3)
# The FC code naming convention here: "FC-STATE-ZIP3" but we store keys as "STATE-ZIP3" for brevity.
NETWORKS = {
    "1FC": {
        "GA-303": 303,
    },
    "4FC": {
        "GA-303": 303,
        "NY-134": 134,
        "TX-799": 799,
        "UT-841": 841,
    },
    "15FC": {
        "AZ-852": 852,
        "CA-900": 900,
        "CA-945": 945,
        "CO-802": 802,
        "FL-331": 331,
        "GA-303": 303,
        "IL-606": 606,
        "MA-021": 21,   # note: 021 -> integer 21 in CSV context if leading zeros dropped
        "MI-481": 481,
        "NC-275": 275,
        "NJ-070": 70,   # 070 -> 70
        "TX-750": 750,
        "TX-770": 770,
        "UT-841": 841,
        "WA-980": 980,
    }
}

# Distance buckets (edges) and labels
BUCKET_EDGES = [-np.inf, 50, 150, 300, 600, 1000, 1400, 1800, np.inf]
BUCKET_LABELS = ["<50","51-150","151-300","301-600","601-1000","1001-1400","1401-1800",">1800"]

def bucketize_distance(d):
    """Map scalar distance to the discrete bucket label."""
    # Using pd.cut later for vectors, but keep helper for robustness
    idx = np.digitize([d], BUCKET_EDGES)[0] - 1
    return BUCKET_LABELS[idx]

# -----------------------------
# Load data
# -----------------------------
zip3_pmf = pd.read_csv(BASE_DIR / "zip3_pmf.csv")
zip3_market = pd.read_csv(BASE_DIR / "zip3_market.csv")
zip3_coords = pd.read_csv(BASE_DIR / "zip3_coordinates.csv")
dist = pd.read_csv(BASE_DIR / "fc_zip3_distance.csv")

# Normalize column names
zip3_pmf.columns = [c.strip().lower() for c in zip3_pmf.columns]
zip3_market.columns = [c.strip().lower() for c in zip3_market.columns]
zip3_coords.columns = [c.strip().lower() for c in zip3_coords.columns]
dist.columns = [c.strip().lower() for c in dist.columns]

# Expected columns (robust handling)
# zip3_pmf: 'zip3', 'pmf'  (if different names, try to infer)
if "zip3" not in zip3_pmf.columns:
    # try 'zip' or 'zip_3' variants
    for alt in ["zip", "zip_3", "zip3_code"]:
        if alt in zip3_pmf.columns:
            zip3_pmf = zip3_pmf.rename(columns={alt: "zip3"})
            break
if "pmf" not in zip3_pmf.columns:
    # guess the name (e.g., 'demand_share')
    for alt in ["demand_share", "prob", "weight"]:
        if alt in zip3_pmf.columns:
            zip3_pmf = zip3_pmf.rename(columns={alt: "pmf"})
            break

# zip3_market: 'zip3', 'market_type'
if "zip3" not in zip3_market.columns:
    for alt in ["zip", "zip_3", "zip3_code"]:
        if alt in zip3_market.columns:
            zip3_market = zip3_market.rename(columns={alt: "zip3"})
            break
if "market_type" not in zip3_market.columns:
    for alt in ["market", "type", "markettype"]:
        if alt in zip3_market.columns:
            zip3_market = zip3_market.rename(columns={alt: "market_type"})
            break

# zip3_coords: 'zip3', 'lat', 'lon' (sometimes 'longitude'/'latitude')
if "zip3" not in zip3_coords.columns:
    for alt in ["zip", "zip_3", "zip3_code"]:
        if alt in zip3_coords.columns:
            zip3_coords = zip3_coords.rename(columns={alt: "zip3"})
            break
if "lat" not in zip3_coords.columns:
    for alt in ["latitude", "y"]:
        if alt in zip3_coords.columns:
            zip3_coords = zip3_coords.rename(columns={alt: "lat"})
            break
if "lon" not in zip3_coords.columns:
    for alt in ["lng", "long", "longitude", "x"]:
        if alt in zip3_coords.columns:
            zip3_coords = zip3_coords.rename(columns={alt: "lon"})
            break

# dist: expect columns: 'fc_zip3', 'zip3', 'distance_miles' (or similar)
# Make it tidy if provided wide; try to infer
# Common shape: columns like fc_303, fc_134... If so, melt long.
if "distance_miles" not in dist.columns:
    # Heuristic: look for 'fc_zip3' & 'zip3' style long format first
    long_has = ("fc_zip3" in dist.columns) and ("zip3" in dist.columns)
    if long_has:
        # Try to find a distance column
        for cand in ["distance", "miles", "dist", "distance_mi", "mi"]:
            if cand in dist.columns:
                dist = dist.rename(columns={cand: "distance_miles"})
                break
    else:
        # Assume wide format: first column is zip3; others are FC zip3s
        # E.g., columns: ['zip3','303','134',...]
        # We'll melt to long: (zip3, fc_zip3, distance_miles)
        # Try to identify the zip3 column
        id_col = None
        for cand in ["zip3","zip","zip_3","zip3_code"]:
            if cand in dist.columns:
                id_col = cand
                break
        if id_col is None:
            raise ValueError("Cannot infer zip3 id column in fc_zip3_distance.csv")
        id_vals = dist[id_col]
        value_cols = [c for c in dist.columns if c != id_col]
        dist_long = dist.melt(id_vars=[id_col], value_vars=value_cols,
                              var_name="fc_zip3", value_name="distance_miles")
        dist_long = dist_long.rename(columns={id_col: "zip3"})
        # Coerce to numeric where possible
        dist_long["fc_zip3"] = dist_long["fc_zip3"].astype(str).str.extract(r'(\d+)').astype(int)
        dist = dist_long

# Coerce types
zip3_pmf["zip3"] = zip3_pmf["zip3"].astype(int)
zip3_market["zip3"] = zip3_market["zip3"].astype(int)
zip3_coords["zip3"] = zip3_coords["zip3"].astype(int)
dist["zip3"] = dist["zip3"].astype(int)
dist["fc_zip3"] = dist["fc_zip3"].astype(int)
dist["distance_miles"] = pd.to_numeric(dist["distance_miles"], errors="coerce")

# Merge base frame
base = zip3_pmf.merge(zip3_market, on="zip3", how="inner").merge(zip3_coords, on="zip3", how="left")

# Utility: ensure pmf sums to 1 (if it's a share). If it's not, we won't force it, but report.
pmf_sum = base["pmf"].sum()
if 0.99 <= pmf_sum <= 1.01:
    base["pmf_norm"] = base["pmf"] / pmf_sum
else:
    # Leave as-is but provide normalized column for proportional uses
    base["pmf_norm"] = base["pmf"] / pmf_sum

def assign_preferred_fc(network_dict):
    """Return assignment DataFrame for a given network (dict: fc_name -> zip3)."""
    fc_zip3s = list(network_dict.values())
    # Filter distance table to only these FCs
    dsub = dist[dist["fc_zip3"].isin(fc_zip3s)].copy()
    # Pivot to (zip3 rows) with each FC as column for fast argmin
    wide = dsub.pivot_table(index="zip3", columns="fc_zip3", values="distance_miles", aggfunc="min")
    # Find nearest FC zip3 per row
    nearest_fc_zip3 = wide.idxmin(axis=1)
    nearest_dist = wide.min(axis=1)
    # Build mapping from fc_zip3 to fc_name
    inv_map = {zip3: name for name, zip3 in network_dict.items()}
    nearest_fc_name = nearest_fc_zip3.map(inv_map)
    # Merge back to base
    assign = base.merge(nearest_fc_zip3.rename("preferred_fc_zip3"), on="zip3", how="left") \
                 .merge(nearest_dist.rename("preferred_fc_distance"), on="zip3", how="left")
    assign["preferred_fc"] = assign["preferred_fc_zip3"].map(inv_map)
    # Distance bucket
    assign["distance_bucket"] = pd.cut(assign["preferred_fc_distance"],
                                       bins=[-np.inf,50,150,300,600,1000,1400,1800,np.inf],
                                       labels=["<50","51-150","151-300","301-600","601-1000","1001-1400","1401-1800",">1800"],
                                       right=True, include_lowest=True)
    return assign

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_clusters(assign, network_name):
    """(a) Scatter plot of ZIP3 points colored by preferred FC with US mainland map as background."""
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.LambertConformal())
    # 미국 본토만 보이도록 영역 설정 (Exclude Alaska, Hawaii)
    ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())

    # 지도 피처 추가
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)

    # ZIP 클러스터 점찍기
    for fc_name, grp in assign.groupby("preferred_fc"):
        ax.scatter(
            grp["lon"], grp["lat"],
            s=6, label=fc_name, alpha=0.7,
            transform=ccrs.PlateCarree()
        )

    # FC 마커 표시
    fc_map = assign.dropna(subset=["preferred_fc_zip3"])\
                   .groupby("preferred_fc")["preferred_fc_zip3"].first().to_dict()
    for fc_name, fc_zip in fc_map.items():
        row = base[base["zip3"] == int(fc_zip)]
        if len(row) > 0:
            ax.scatter(
                float(row["lon"].iloc[0]), float(row["lat"].iloc[0]),
                s=100, marker="^", edgecolor="k", linewidths=0.5,
                label=f"{fc_name} (FC)",
                transform=ccrs.PlateCarree()
            )

    ax.legend(markerscale=2, fontsize=7, ncol=2, frameon=False, loc="lower left")
    ax.set_title(f"Task 3a — ZIP3 Clusters by Preferred FC ({network_name})", fontsize=12)

    out_path = OUT_DIR / f"task3a_{network_name}_map.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def compute_task3b(assign, network_name):
    """(b) Aggregated demand share per FC and per (FC × market type)."""
    # Using normalized pmf for shares
    g_fc = assign.groupby("preferred_fc")["pmf_norm"].sum().reset_index().rename(columns={"pmf_norm":"demand_share"})
    g_fc["network"] = network_name
    g_fc.to_csv(OUT_DIR / f"task3b_fc_demand_share_{network_name}.csv", index=False)

    g_fc_mt = assign.groupby(["preferred_fc","market_type"])["pmf_norm"].sum().reset_index() \
                    .rename(columns={"pmf_norm":"demand_share"})
    g_fc_mt["network"] = network_name
    g_fc_mt.to_csv(OUT_DIR / f"task3b_fc_market_demand_share_{network_name}.csv", index=False)
    return g_fc, g_fc_mt

def compute_task3c(assign, network_name):
    """(c) Demand distribution over distance buckets per market type.
       Also returns optional granular FC×market×bucket distribution.
    """
    # per market type × bucket
    mt_bucket = assign.groupby(["market_type","distance_bucket"])["pmf_norm"].sum().reset_index() \
                      .rename(columns={"pmf_norm":"demand_share"})
    mt_bucket["network"] = network_name
    mt_bucket.to_csv(OUT_DIR / f"task3c_market_distance_distribution_{network_name}.csv", index=False)

    # optional: FC × market type × bucket (can help later tasks)
    fc_mt_bucket = assign.groupby(["preferred_fc","market_type","distance_bucket"])["pmf_norm"].sum().reset_index() \
                          .rename(columns={"pmf_norm":"demand_share"})
    fc_mt_bucket["network"] = network_name
    fc_mt_bucket.to_csv(OUT_DIR / f"task3c_fc_market_distance_distribution_{network_name}.csv", index=False)

    return mt_bucket, fc_mt_bucket

def plot_task3c(mt_bucket, network_name):
    """Simple bar chart per market showing distribution by buckets (one figure per market)."""
    markets = mt_bucket["market_type"].dropna().unique().tolist()
    for m in markets:
        sub = mt_bucket[mt_bucket["market_type"] == m].copy()
        # Ensure bucket order
        cats = ["<50","51-150","151-300","301-600","601-1000","1001-1400","1401-1800",">1800"]
        sub["distance_bucket"] = pd.Categorical(sub["distance_bucket"], categories=cats, ordered=True)
        sub = sub.sort_values("distance_bucket")
        fig, ax = plt.subplots(figsize=(9,4.5))
        ax.bar(sub["distance_bucket"].astype(str), sub["demand_share"])
        ax.set_title(f"Task 3c — {m} demand distribution by distance bucket ({network_name})")
        ax.set_xlabel("Distance bucket (miles)")
        ax.set_ylabel("Demand share (PMF)")
        fig.tight_layout()
        out_path = OUT_DIR / f"task3c_{network_name}_{m}_bar.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

def run_for_network(network_name, fc_dict):
    assign = assign_preferred_fc(fc_dict)
    # Save assignment for audit
    assign[["zip3","preferred_fc","preferred_fc_zip3","preferred_fc_distance","distance_bucket","market_type","pmf","pmf_norm","lat","lon"]]\
        .to_csv(OUT_DIR / f"assignment_{network_name}.csv", index=False)

    # (a) plot clusters
    plot_clusters(assign, network_name)

    # (b) demand shares
    g_fc, g_fc_mt = compute_task3b(assign, network_name)

    # (c) distance distributions
    mt_bucket, fc_mt_bucket = compute_task3c(assign, network_name)
    plot_task3c(mt_bucket, network_name)

def main():
    for net, fcs in NETWORKS.items():
        run_for_network(net, fcs)

if __name__ == "__main__":
    main()
