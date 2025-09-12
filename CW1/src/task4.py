# task4_full.py
# Dobda Case — Task 4 (a,b,c,d) Full Automation (no numeric answers printed)
# - Generates CSVs and plots under output/task4
# - Comments in English; avoids printing final numbers to respect homework policy

from __future__ import annotations
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: Cartopy map background
_HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    _HAS_CARTOPY = False

# -----------------------------
# Configuration
# -----------------------------
def resolve_paths():
    """
    Resolve project paths robustly:
    - Default: project/CW1 (script is under project/CW1/scripts or similar)
    - Fallback to current working dir if relative layout is different.
    """
    here = Path(__file__).resolve()
    # Try .../project/CW1/
    try_base = here.parents[1]
    data = try_base / "data"
    out = try_base / "output" / "task4"
    if data.exists():
        return try_base, data, out
    # Fallback: current dir has data/
    cwd = Path.cwd()
    if (cwd / "data").exists():
        return cwd, cwd / "data", cwd / "output" / "task4"
    # Final fallback: script's dir has data/
    if (here.parent / "data").exists():
        return here.parent, here.parent / "data", here.parent / "output" / "task4"
    # If nothing found, just assume cwd/data
    return cwd, cwd / "data", cwd / "output" / "task4"

BASE_DIR, DATA_DIR, OUT_DIR = resolve_paths()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fulfillment networks
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

# Distance buckets (Task 3 & 4 spec)
BUCKET_EDGES = [-np.inf, 50, 150, 300, 600, 1000, 1400, 1800, np.inf]
BUCKET_LABELS = ["<50","51-150","151-300","301-600","601-1000","1001-1400","1401-1800",">1800"]
BUCKET_ORD = {lab:i for i,lab in enumerate(BUCKET_LABELS)}

# -----------------------------
# Utilities
# -----------------------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def bucketize_series(dist_series: pd.Series) -> pd.Categorical:
    return pd.cut(
        dist_series, bins=BUCKET_EDGES, labels=BUCKET_LABELS,
        right=True, include_lowest=True
    )

def bucket_next(label: str) -> str | None:
    """Return the next higher bucket label, or None if already the highest."""
    i = BUCKET_ORD[label]
    if i + 1 < len(BUCKET_LABELS):
        return BUCKET_LABELS[i + 1]
    return None

def _colormap_by_category(keys: pd.Series) -> dict:
    """
    Deterministic color mapping for many categories:
    - Hash category to pick a color index from a qualitative palette.
    """
    import hashlib
    cmap = plt.get_cmap("tab20")
    out = {}
    for k in keys.unique():
        h = int(hashlib.md5(str(k).encode("utf-8")).hexdigest(), 16)
        out[k] = cmap(h % cmap.N)
    return out

def _colormap_red_yellow_green(values: pd.Series) -> tuple[plt.cm.ScalarMappable, tuple[float, float]]:
    """
    Build a red-yellow-green scalar mappable for continuous-ish values.
    Returns (mappable, (vmin, vmax))
    """
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:  # avoid zero range
        vmin, vmax = float(values.min()) - 1, float(values.max()) + 1
    cmap = plt.get_cmap("RdYlGn")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm, (vmin, vmax)

# -----------------------------
# Data Loading
# -----------------------------
def load_data():
    zip3_pmf = _norm_cols(pd.read_csv(DATA_DIR / "zip3_pmf.csv"))
    zip3_market = _norm_cols(pd.read_csv(DATA_DIR / "zip3_market.csv"))
    zip3_coords = _norm_cols(pd.read_csv(DATA_DIR / "zip3_coordinates.csv"))
    dist = _norm_cols(pd.read_csv(DATA_DIR / "fc_zip3_distance.csv"))

    # Normalize column names and types
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

    # Distance table to long format if needed
    if "distance_miles" not in dist.columns:
        id_col = [c for c in dist.columns if "zip" in c][0]
        value_cols = [c for c in dist.columns if c != id_col]
        dist = dist.melt(
            id_vars=[id_col], value_vars=value_cols,
            var_name="fc_zip3", value_name="distance_miles"
        )
        dist = dist.rename(columns={id_col: "zip3"})
        dist["fc_zip3"] = dist["fc_zip3"].astype(str).str.extract(r'(\d+)').astype(int)

    # Types
    for df in [zip3_pmf, zip3_market, zip3_coords]:
        df["zip3"] = df["zip3"].astype(int)
    dist["zip3"] = dist["zip3"].astype(int)
    dist["fc_zip3"] = dist["fc_zip3"].astype(int)
    dist["distance_miles"] = pd.to_numeric(dist["distance_miles"], errors="coerce")

    # Base frame with PMF normalization
    base = zip3_pmf.merge(zip3_market, on="zip3").merge(zip3_coords, on="zip3", how="left")
    base["pmf_norm"] = base["pmf"] / base["pmf"].sum()
    return base, dist

# -----------------------------
# Task 3 Base (Preferred FC)
# -----------------------------
def compute_preferred_assignment(base: pd.DataFrame, dist: pd.DataFrame, network_dict: dict) -> pd.DataFrame:
    """Nearest (preferred) FC distance and bucket per ZIP (Task 3 core used by Task 4)."""
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
    return assign, dsub, inv_map

# -----------------------------
# Task 4 (a,b,c,d)
# -----------------------------
def build_multisource_clusters(assign: pd.DataFrame, dsub: pd.DataFrame, inv_map: dict) -> pd.DataFrame:
    """
    For each ZIP, find the set of FCs whose distance bucket is within:
      - the same bucket as the preferred FC distance, OR
      - the next higher bucket.
    Returns 'assign2' with added columns:
      - candidate_fc_zip3_set (frozenset of int)
      - candidate_fc_name_set (frozenset of str)
      - candidate_count (int)
      - cluster_id (stable string key)
    """
    # Precompute per-zip all FC distances
    per_zip = dsub.pivot_table(index="zip3", columns="fc_zip3", values="distance_miles", aggfunc="min")
    per_zip = per_zip.reindex(assign["zip3"].unique())  # align

    # Bucketize every (zip, fc) distance into labels
    dist_long = per_zip.reset_index().melt(id_vars="zip3", var_name="fc_zip3", value_name="distance_miles")
    dist_long["bucket"] = bucketize_series(dist_long["distance_miles"])

    # Join preferred bucket
    pref = assign[["zip3", "preferred_bucket"]].drop_duplicates()
    dist_long = dist_long.merge(pref, on="zip3", how="left", suffixes=("", "_pref"))

    # Compute allowed buckets per ZIP: preferred and next
    next_bucket_map = {lab: bucket_next(lab) for lab in BUCKET_LABELS}
    allowed = {}
    for _, r in pref.iterrows():
        b0 = r["preferred_bucket"]
        b1 = next_bucket_map[str(b0)] if pd.notna(b0) else None
        allowed[r["zip3"]] = {b0, b1} if b1 is not None else {b0}

    # Keep candidates within allowed buckets
    def _is_allowed(row) -> bool:
        z = row["zip3"]
        b = row["bucket"]
        return b in allowed.get(z, set())

    dist_long["is_allowed"] = dist_long.apply(_is_allowed, axis=1)

    cand = dist_long[dist_long["is_allowed"] & dist_long["bucket"].notna()].copy()
    # Aggregate candidate sets per ZIP
    agg = cand.groupby("zip3").agg(
        candidate_fc_zip3_set=("fc_zip3", lambda s: frozenset(s.dropna().astype(int))),
    ).reset_index()
    # Map to names
    def map_names(fs: frozenset) -> frozenset:
        return frozenset(inv_map.get(int(z), f"ZIP{int(z)}") for z in fs)
    agg["candidate_fc_name_set"] = agg["candidate_fc_zip3_set"].apply(map_names)
    agg["candidate_count"] = agg["candidate_fc_zip3_set"].apply(lambda s: len(s))

    # Add stable cluster id (sorted FC names joined)
    def make_cluster_id(fs: frozenset) -> str:
        return "|".join(sorted(list(fs)))
    agg["cluster_id"] = agg["candidate_fc_name_set"].apply(make_cluster_id)

    # Merge back to assign
    assign2 = assign.merge(agg, on="zip3", how="left")

    # For ZIPs that lacked any candidate (edge cases), default to preferred-only
    no_cand = assign2["candidate_count"].isna()
    if no_cand.any():
        warnings.warn(f"{no_cand.sum()} ZIPs had no candidate set; defaulting to preferred-only.")
        assign2.loc[no_cand, "candidate_fc_zip3_set"] = assign2.loc[no_cand, "preferred_fc_zip3"].apply(lambda z: frozenset([int(z)]) if pd.notna(z) else frozenset())
        assign2.loc[no_cand, "candidate_fc_name_set"] = assign2.loc[no_cand, "preferred_fc"].apply(lambda n: frozenset([n]) if pd.notna(n) else frozenset())
        assign2.loc[no_cand, "candidate_count"] = 1
        assign2.loc[no_cand, "cluster_id"] = assign2.loc[no_cand, "candidate_fc_name_set"].apply(lambda fs: "|".join(sorted(list(fs))))

    return assign2

def plot_clusters(assign2: pd.DataFrame, network_name: str, out_dir: Path):
    """
    (a) Geographical plot colored by fulfillment clusters (same FC set).
    """
    df = assign2.dropna(subset=["lat","lon","cluster_id"]).copy()
    color_map = _colormap_by_category(df["cluster_id"])
    colors = df["cluster_id"].map(color_map)

    title = f"Task 4a — Fulfillment Clusters (same/next bucket candidates) — {network_name}"
    out_path = out_dir / f"task4a_{network_name}_clusters_map.png"

    if _HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10,6))
        ax = plt.axes(projection=proj)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
        ax.set_extent([-125, -66.5, 24, 49], crs=proj)  # CONUS
        ax.scatter(df["lon"], df["lat"], s=6, c=colors, transform=proj)
        ax.set_title(title)
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(df["lon"], df["lat"], s=6, c=colors)
        ax.set_title(title)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

def plot_fc_count_heat(assign2: pd.DataFrame, network_name: str, out_dir: Path):
    """
    (b) Plot colored by number of FCs that can serve the cluster (red=low, green=high).
    """
    df = assign2.dropna(subset=["lat","lon","candidate_count"]).copy()
    sm, (vmin, vmax) = _colormap_red_yellow_green(df["candidate_count"])
    colors = sm.to_rgba(df["candidate_count"].values)

    title = f"Task 4b — Candidate FC Count per ZIP (red=low → green=high) — {network_name}"
    out_path = out_dir / f"task4b_{network_name}_fc_count_map.png"

    if _HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10,6))
        ax = plt.axes(projection=proj)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
        ax.set_extent([-125, -66.5, 24, 49], crs=proj)
        sc = ax.scatter(df["lon"], df["lat"], s=6, c=df["candidate_count"], cmap="RdYlGn", transform=proj, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("# of candidate FCs")
        ax.set_title(title)
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8,6))
        sc = ax.scatter(df["lon"], df["lat"], s=6, c=df["candidate_count"], cmap="RdYlGn", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label("# of candidate FCs")
        ax.set_title(title)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

def compute_single_fc_proportion(assign2: pd.DataFrame) -> pd.DataFrame:
    """
    (c) Proportion of demand that can only be served by a single FC.
    Use PMF-normalized weights (pmf_norm).
    Returns a one-row DataFrame with columns: total_pmf, single_pmf, proportion
    """
    df = assign2[["pmf_norm", "candidate_count"]].copy()
    df = df.dropna(subset=["pmf_norm","candidate_count"])
    total = df["pmf_norm"].sum()
    single = df.loc[df["candidate_count"] == 1, "pmf_norm"].sum()
    prop = single / total if total > 0 else np.nan
    return pd.DataFrame({"total_pmf":[total], "single_pmf":[single], "proportion":[prop]})

def _distance_bucket_for(zip3: int, fc_zip3: int, dsub: pd.DataFrame) -> str:
    """
    Helper: Get the distance bucket label for a specific (ZIP, FC) pair.
    """
    v = dsub[(dsub["zip3"] == zip3) & (dsub["fc_zip3"] == fc_zip3)]["distance_miles"]
    if v.empty:
        return np.nan
    return bucketize_series(pd.Series([float(v.iloc[0])])).iloc[0]

def compute_bucket_distributions(assign2: pd.DataFrame, dsub: pd.DataFrame, inv_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (d) Distance-bucket demand distributions under:
      - closest-only (baseline)
      - 90/10 reallocation among candidate FCs (Task 4 rule)
    Returns (baseline_bucket_dist, reallocated_bucket_dist)
    Both are DataFrames with:
      columns: ['bucket','demand_share'] where demand_share sums to 1 over buckets
    """
    # --- Baseline: 100% to preferred FC, bucket by preferred distance
    base = assign2.dropna(subset=["pmf_norm","preferred_bucket"]).copy()
    baseline = base.groupby("preferred_bucket", as_index=False)["pmf_norm"].sum()
    baseline = baseline.rename(columns={"preferred_bucket":"bucket","pmf_norm":"demand"})
    # Ensure all buckets present
    baseline = baseline.set_index("bucket").reindex(BUCKET_LABELS, fill_value=0.0).reset_index()
    baseline["demand_share"] = baseline["demand"] / baseline["demand"].sum()

    # --- Reallocated: 90% preferred, 10% spread equally to other candidates in the same cluster
    rows = []
    for _, r in assign2.iterrows():
        pmf = r.get("pmf_norm", np.nan)
        if not pd.notna(pmf) or pmf <= 0:
            continue
        pref_fc_zip3 = r.get("preferred_fc_zip3", np.nan)
        cand_set = r.get("candidate_fc_zip3_set", frozenset())
        if not pd.notna(pref_fc_zip3):
            continue

        # 90% to preferred FC's bucket
        b_pref = _distance_bucket_for(int(r["zip3"]), int(pref_fc_zip3), dsub)
        if pd.notna(b_pref):
            rows.append((b_pref, 0.9 * pmf))

        # 10% split to others (if any)
        others = [int(z) for z in cand_set if int(z) != int(pref_fc_zip3)]
        k = len(others)
        if k > 0:
            share = 0.1 * pmf / k
            for z in others:
                b = _distance_bucket_for(int(r["zip3"]), z, dsub)
                if pd.notna(b):
                    rows.append((b, share))
        else:
            # If no others, route 100% to preferred to preserve mass
            rows[-1] = (rows[-1][0], rows[-1][1] + 0.1 * pmf)  # add the leftover to pref

    realloc = pd.DataFrame(rows, columns=["bucket","demand"])
    if realloc.empty:
        realloc = pd.DataFrame({"bucket": BUCKET_LABELS, "demand": [0.0]*len(BUCKET_LABELS)})
    else:
        realloc = realloc.groupby("bucket", as_index=False)["demand"].sum()
        realloc = realloc.set_index("bucket").reindex(BUCKET_LABELS, fill_value=0.0).reset_index()

    realloc["demand_share"] = realloc["demand"] / realloc["demand"].sum() if realloc["demand"].sum() > 0 else 0.0
    return baseline[["bucket","demand_share"]], realloc[["bucket","demand_share"]]

def save_tables_and_plots(network_name: str, assign2: pd.DataFrame, dsub: pd.DataFrame, out_dir: Path):
    # Save enriched ZIP-level assignment with candidate sets (serialized)
    z = assign2.copy()
    z["candidate_fc_zip3_set"] = z["candidate_fc_zip3_set"].apply(lambda s: ",".join(map(str, sorted(list(s)))) if isinstance(s, frozenset) else "")
    z["candidate_fc_name_set"] = z["candidate_fc_name_set"].apply(lambda s: ",".join(sorted(list(s))) if isinstance(s, frozenset) else "")
    z.to_csv(out_dir / f"task4_{network_name}_zip_assignment.csv", index=False)

    # Cluster summary (cluster_id, size, pmf share)
    clus = assign2.groupby(["cluster_id"], as_index=False).agg(
        cluster_size=("candidate_count","first"),
        zip_count=("zip3","count"),
        pmf_share=("pmf_norm","sum")
    )
    clus = clus.sort_values(["cluster_size","zip_count"], ascending=[True, False])
    clus.to_csv(out_dir / f"task4_{network_name}_cluster_summary.csv", index=False)

    # (c) Single-FC feasible proportion
    prop = compute_single_fc_proportion(assign2)
    prop.to_csv(out_dir / f"task4_{network_name}_single_fc_proportion.csv", index=False)

    # (d) Distance-bucket distributions (baseline vs reallocated)
    baseline_dist, realloc_dist = compute_bucket_distributions(assign2, dsub, {})
    baseline_dist.to_csv(out_dir / f"task4_{network_name}_bucket_distribution_baseline.csv", index=False)
    realloc_dist.to_csv(out_dir / f"task4_{network_name}_bucket_distribution_reallocated.csv", index=False)

    # Small comparison plot (shares by bucket)
    fig, ax = plt.subplots(figsize=(9,4.5))
    ax.plot(baseline_dist["bucket"], baseline_dist["demand_share"], marker="o", label="Closest-only")
    ax.plot(realloc_dist["bucket"], realloc_dist["demand_share"], marker="s", label="90/10 reallocated")
    ax.set_title(f"Task 4d — Distance-bucket demand share (baseline vs 90/10) — {network_name}")
    ax.set_xlabel("Distance bucket (miles)")
    ax.set_ylabel("Demand share")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / f"task4d_{network_name}_bucket_share_comparison.png", dpi=220)
    plt.close(fig)

def run_for_network(network_name: str, network_dict: dict):
    print(f"▶ Running Task 4 for network {network_name} ...")
    base, dist = load_data()
    assign, dsub, inv_map = compute_preferred_assignment(base, dist, network_dict)
    assign2 = build_multisource_clusters(assign, dsub, inv_map)

    # (a) cluster map
    plot_clusters(assign2, network_name, OUT_DIR)
    # (b) fc-count heat map
    plot_fc_count_heat(assign2, network_name, OUT_DIR)
    # Save tables and (c)(d) outputs
    save_tables_and_plots(network_name, assign2, dsub, OUT_DIR)

    print(f"  ✓ Outputs saved under: {OUT_DIR}")

# -----------------------------
# CLI
# -----------------------------
def main():
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", OUT_DIR)
    parser = argparse.ArgumentParser(description="Dobda Task 4 (a,b,c,d) full automation")
    parser.add_argument("--network", type=str, default="ALL", choices=["ALL","1FC","4FC","15FC"],
                        help="Run for a specific network or ALL")
    args = parser.parse_args()

    if args.network == "ALL":
        for net, fcs in NETWORKS.items():
            run_for_network(net, fcs)
    else:
        run_for_network(args.network, NETWORKS[args.network])

if __name__ == "__main__":
    main()
