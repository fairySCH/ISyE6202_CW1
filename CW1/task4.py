# task4_solution.py
# Dobda Case — Task 4 (a,b,c,d) Automation
# Author: ChatGPT
#
# Inputs (same directory):
#   - zip3_pmf.csv
#   - zip3_market.csv
#   - zip3_coordinates.csv
#   - fc_zip3_distance.csv
#   - (optional) dobda_task3_outputs/assignment_{NETWORK}.csv
#
# Outputs: dobda_task4_outputs/
#   (a) task4a_{NETWORK}_clusters_map.png
#   (b) task4b_{NETWORK}_fc_count_map.png
#   (c) task4c_{NETWORK}_single_fc_only.csv / .txt
#   (d) task4d_{NETWORK}_market_distance_distribution.csv + per-market bar charts
#   helper: task4_assignment_{NETWORK}.csv, task4_demand_split_{NETWORK}.csv

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try Cartopy (map background). If not available, fall back to plain scatter.
_HAS_CARTOPY = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    _HAS_CARTOPY = False

# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path("./data")  # <-- 데이터 파일이 있는 폴더 (필요시 수정)
OUT_DIR = BASE_DIR / "dobda_task4_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORKS = {
    "1FC": {"GA-303": 303},
    "4FC": {"GA-303": 303, "NY-134": 134, "TX-799": 799, "UT-841": 841},
    "15FC": {
        "AZ-852": 852, "CA-900": 900, "CA-945": 945, "CO-802": 802,
        "FL-331": 331, "GA-303": 303, "IL-606": 606, "MA-021": 21,
        "MI-481": 481, "NC-275": 275, "NJ-070": 70, "TX-750": 750,
        "TX-770": 770, "UT-841": 841, "WA-980": 980
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

zip3_pmf = _norm_cols(pd.read_csv(BASE_DIR / "zip3_pmf.csv"))
zip3_market = _norm_cols(pd.read_csv(BASE_DIR / "zip3_market.csv"))
zip3_coords = _norm_cols(pd.read_csv(BASE_DIR / "zip3_coordinates.csv"))
dist = _norm_cols(pd.read_csv(BASE_DIR / "fc_zip3_distance.csv"))

# Normalize column names/types
if "zip3" not in zip3_pmf.columns:
    for alt in ["zip", "zip_3", "zip3_code"]:
        if alt in zip3_pmf.columns:
            zip3_pmf = zip3_pmf.rename(columns={alt:"zip3"}); break
if "pmf" not in zip3_pmf.columns:
    for alt in ["demand_share","prob","weight"]:
        if alt in zip3_pmf.columns:
            zip3_pmf = zip3_pmf.rename(columns={alt:"pmf"}); break

if "zip3" not in zip3_market.columns:
    for alt in ["zip", "zip_3", "zip3_code"]:
        if alt in zip3_market.columns:
            zip3_market = zip3_market.rename(columns={alt:"zip3"}); break
if "market_type" not in zip3_market.columns:
    for alt in ["market","type","markettype"]:
        if alt in zip3_market.columns:
            zip3_market = zip3_market.rename(columns={alt:"market_type"}); break

if "zip3" not in zip3_coords.columns:
    for alt in ["zip","zip_3","zip3_code"]:
        if alt in zip3_coords.columns:
            zip3_coords = zip3_coords.rename(columns={alt:"zip3"}); break
if "lat" not in zip3_coords.columns:
    for alt in ["latitude","y"]:
        if alt in zip3_coords.columns:
            zip3_coords = zip3_coords.rename(columns={alt:"lat"}); break
if "lon" not in zip3_coords.columns:
    for alt in ["lng","long","longitude","x"]:
        if alt in zip3_coords.columns:
            zip3_coords = zip3_coords.rename(columns={alt:"lon"}); break

# Distance tidy (long format)
if "distance_miles" not in dist.columns:
    long_has = ("fc_zip3" in dist.columns) and ("zip3" in dist.columns)
    if long_has:
        for cand in ["distance","miles","dist","distance_mi","mi"]:
            if cand in dist.columns:
                dist = dist.rename(columns={cand:"distance_miles"}); break
    else:
        id_col = None
        for cand in ["zip3","zip","zip_3","zip3_code"]:
            if cand in dist.columns: id_col = cand; break
        if id_col is None:
            raise ValueError("Cannot infer zip3 id column in fc_zip3_distance.csv")
        value_cols = [c for c in dist.columns if c != id_col]
        dist_long = dist.melt(id_vars=[id_col], value_vars=value_cols,
                              var_name="fc_zip3", value_name="distance_miles")
        dist_long = dist_long.rename(columns={id_col:"zip3"})
        dist_long["fc_zip3"] = dist_long["fc_zip3"].astype(str).str.extract(r'(\d+)').astype(int)
        dist = dist_long

# Types
zip3_pmf["zip3"] = zip3_pmf["zip3"].astype(int)
zip3_market["zip3"] = zip3_market["zip3"].astype(int)
zip3_coords["zip3"] = zip3_coords["zip3"].astype(int)
dist["zip3"] = dist["zip3"].astype(int)
dist["fc_zip3"] = dist["fc_zip3"].astype(int)
dist["distance_miles"] = pd.to_numeric(dist["distance_miles"], errors="coerce")

# Base frame + normalized pmf
base = zip3_pmf.merge(zip3_market, on="zip3", how="inner").merge(zip3_coords, on="zip3", how="left")
pmf_sum = base["pmf"].sum()
base["pmf_norm"] = base["pmf"] / pmf_sum

# -----------------------------
# Preferred assignment (Task 3 base)
# -----------------------------
def compute_preferred_assignment(network_dict):
    fc_zip3s = list(network_dict.values())
    dsub = dist[dist["fc_zip3"].isin(fc_zip3s)].copy()
    wide = dsub.pivot_table(index="zip3", columns="fc_zip3", values="distance_miles", aggfunc="min")
    nearest_fc_zip3 = wide.idxmin(axis=1)
    nearest_dist = wide.min(axis=1)
    inv_map = {zip3: name for name, zip3 in network_dict.items()}
    assign = base.merge(nearest_fc_zip3.rename("preferred_fc_zip3"), on="zip3", how="left") \
                 .merge(nearest_dist.rename("preferred_fc_distance"), on="zip3", how="left")
    assign["preferred_fc"] = assign["preferred_fc_zip3"].map(inv_map)
    assign["preferred_bucket"] = bucketize_series(assign["preferred_fc_distance"])
    return assign

def load_or_compute_assignment(network_name, network_dict):
    t3_path = BASE_DIR / "dobda_task3_outputs" / f"assignment_{network_name}.csv"
    if t3_path.exists():
        assign = pd.read_csv(t3_path)
        if "preferred_bucket" not in assign.columns and "preferred_fc_distance" in assign.columns:
            assign["preferred_bucket"] = bucketize_series(assign["preferred_fc_distance"])
        if "pmf_norm" not in assign.columns and "pmf" in assign.columns:
            assign = assign.drop(columns=[c for c in ["pmf_norm"] if c in assign.columns])
            assign = assign.merge(base[["zip3","pmf_norm"]], on="zip3", how="left")
        if "lat" not in assign.columns or "lon" not in assign.columns:
            assign = assign.merge(base[["zip3","lat","lon"]], on="zip3", how="left")
        if "market_type" not in assign.columns:
            assign = assign.merge(base[["zip3","market_type"]], on="zip3", how="left")
        return assign
    else:
        return compute_preferred_assignment(network_dict)

# -----------------------------
# Task 4 core: Build fulfillment sets
# -----------------------------
def build_fulfillment_sets(assign, network_dict):
    """ZIP별로 preferred 버킷 또는 그 다음 버킷에 속하는 모든 FC를 set으로 수집"""
    fc_zip3s = list(network_dict.values())
    dsub = dist[dist["fc_zip3"].isin(fc_zip3s)].copy()
    dsub["bucket"] = bucketize_series(dsub["distance_miles"])

    pref = assign[["zip3","preferred_bucket","preferred_fc"]].copy()
    merged = dsub.merge(pref, on="zip3", how="left")

    def bucket_ok(row):
        if pd.isna(row["bucket"]) or pd.isna(row["preferred_bucket"]):
            return False
        b = str(row["bucket"]); p = str(row["preferred_bucket"])
        return (BUCKET_ORD.get(b, 99) == BUCKET_ORD.get(p, -1)) or \
               (BUCKET_ORD.get(b, 99) == BUCKET_ORD.get(p, -1) + 1)

    merged["eligible"] = merged.apply(bucket_ok, axis=1)

    inv_map = {zip3: name for name, zip3 in network_dict.items()}
    elig = merged[merged["eligible"]].copy()
    elig["fc_name"] = elig["fc_zip3"].map(inv_map)

    fcset = elig.groupby("zip3")["fc_name"].apply(lambda s: sorted(set(s.dropna().tolist()))).reset_index()
    fcset = fcset.rename(columns={"fc_name":"fc_set"})

    # ensure preferred FC present
    pref_fc_map = assign.set_index("zip3")["preferred_fc"].to_dict()
    def ensure_pref(row):
        s = list(row["fc_set"])
        p = pref_fc_map.get(row["zip3"])
        if p and p not in s: s.append(p)
        return sorted(s)
    fcset["fc_set"] = fcset.apply(ensure_pref, axis=1)
    fcset["fc_count"] = fcset["fc_set"].apply(len)
    fcset["cluster_key"] = fcset["fc_set"].apply(lambda L: "|".join(L))

    key_to_id = {k:i for i,k in enumerate(sorted(fcset["cluster_key"].unique()))}
    fcset["cluster_id"] = fcset["cluster_key"].map(key_to_id)

    out = assign.merge(fcset, on="zip3", how="left")

    # fallback: preferred only
    mask_na = out["fc_set"].isna()
    if mask_na.any():
        out.loc[mask_na, "fc_set"] = out.loc[mask_na, "preferred_fc"].apply(lambda x: [x] if pd.notna(x) else [])
        out.loc[mask_na, "fc_count"] = out.loc[mask_na, "fc_set"].apply(len)
        out.loc[mask_na, "cluster_key"] = out.loc[mask_na, "fc_set"].apply(lambda L: "|".join(L))
        for k in out.loc[mask_na, "cluster_key"].unique():
            if k not in key_to_id:
                key_to_id[k] = max(key_to_id.values(), default=-1) + 1
        out.loc[mask_na, "cluster_id"] = out.loc[mask_na, "cluster_key"].map(key_to_id)

    return out

# -----------------------------
# Plots
# -----------------------------
def _plot_background(ax=None):
    if not _HAS_CARTOPY:
        return None
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5)
    return ax

def plot_clusters_map(assign4, network_name):
    """(a) Fulfillment clusters 지도"""
    if _HAS_CARTOPY:
        fig = plt.figure(figsize=(12,7))
        ax = plt.axes(projection=ccrs.LambertConformal())
        ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
        _plot_background(ax)
        for cid, grp in assign4.groupby("cluster_id"):
            ax.scatter(grp["lon"], grp["lat"], s=5, alpha=0.7,
                       transform=ccrs.PlateCarree())
        ax.set_title(f"Task 4a — Fulfillment Clusters (same/next bucket sets) — {network_name}")
        out_path = OUT_DIR / f"task4a_{network_name}_clusters_map.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        # Fallback: plain scatter
        fig, ax = plt.subplots(figsize=(12,7))
        for cid, grp in assign4.groupby("cluster_id"):
            ax.scatter(grp["lon"], grp["lat"], s=5, alpha=0.7)
        ax.set_title(f"Task 4a — Fulfillment Clusters — {network_name} (NO MAP BG)")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        out_path = OUT_DIR / f"task4a_{network_name}_clusters_map.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

def plot_fc_count_map(assign4, network_name):
    """(b) FC 개수 지도(Red=1, Yellow=2, Green=3+)"""
    # 색상 지정은 요구사항(빨/노/초)에 맞게 지정
    def count_to_color(k):
        if pd.isna(k): return "gray"
        k = int(k)
        if k <= 1: return "red"
        elif k == 2: return "yellow"
        else: return "green"

    assign4 = assign4.copy()
    assign4["count_color"] = assign4["fc_count"].apply(count_to_color)

    if _HAS_CARTOPY:
        fig = plt.figure(figsize=(12,7))
        ax = plt.axes(projection=ccrs.LambertConformal())
        ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
        _plot_background(ax)
        for color, grp in assign4.groupby("count_color"):
            ax.scatter(grp["lon"], grp["lat"], s=5, alpha=0.7, color=color,
                       transform=ccrs.PlateCarree(), label=color)
        ax.set_title(f"Task 4b — # of FCs per ZIP (Red=1, Yellow=2, Green=3+) — {network_name}")
        ax.legend(markerscale=2, fontsize=8, frameon=False, loc="lower left")
        out_path = OUT_DIR / f"task4b_{network_name}_fc_count_map.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(12,7))
        for color, grp in assign4.groupby("count_color"):
            ax.scatter(grp["lon"], grp["lat"], s=5, alpha=0.7, color=color, label=color)
        ax.set_title(f"Task 4b — # of FCs per ZIP — {network_name} (NO MAP BG)")
        ax.legend(markerscale=2, fontsize=8, frameon=False, loc="lower left")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        out_path = OUT_DIR / f"task4b_{network_name}_fc_count_map.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

# -----------------------------
# (c) Single-FC-only share
# -----------------------------
def compute_single_fc_share(assign4, network_name):
    total = assign4["pmf_norm"].sum()
    single = assign4.loc[assign4["fc_count"]==1, "pmf_norm"].sum()
    share = single / total if total > 0 else np.nan
    df = pd.DataFrame([{"network":network_name, "single_fc_demand_share": share}])
    df.to_csv(OUT_DIR / f"task4c_{network_name}_single_fc_only.csv", index=False)
    with open(OUT_DIR / f"task4c_{network_name}_single_fc_only.txt", "w") as f:
        f.write(f"{share:.6f}\n")
    return share

# -----------------------------
# (d) 90/10 split & aggregate
# -----------------------------
def compute_split_distribution(assign4, network_dict, network_name):
    rows = []
    inv_map = {zip3: name for name, zip3 in network_dict.items()}

    dsub = dist[dist["fc_zip3"].isin(network_dict.values())].copy()
    dmin = dsub.groupby(["zip3","fc_zip3"], as_index=False)["distance_miles"].min()
    dmin["bucket"] = bucketize_series(dmin["distance_miles"])
    dmin["fc_name"] = dmin["fc_zip3"].map(inv_map)
    name_to_zip3 = {name:zip3 for name, zip3 in network_dict.items()}

    tmp = assign4[["zip3","pmf_norm","market_type","preferred_fc","fc_set"]].copy()
    tmp = tmp.dropna(subset=["fc_set"])
    tmp["fc_set"] = tmp["fc_set"].apply(lambda L: L if isinstance(L, list) else [])
    tmp["other_fcs"] = tmp.apply(lambda r: [x for x in r["fc_set"] if x != r["preferred_fc"]], axis=1)

    for _, r in tmp.iterrows():
        z = int(r["zip3"]); m = r["market_type"]; p = r["preferred_fc"]; pmf = float(r["pmf_norm"])
        others = r["other_fcs"]
        allocs = [(p, 0.9*pmf)]
        if len(others) > 0:
            split = (0.1*pmf) / len(others)
            for oc in others:
                allocs.append((oc, split))
        for fc_name, q in allocs:
            fcz = name_to_zip3.get(fc_name)
            row = dmin[(dmin["zip3"]==z) & (dmin["fc_zip3"]==fcz)]
            if len(row) == 0:
                bkt = np.nan
            else:
                b = row["bucket"].iloc[0]
                bkt = str(b) if not pd.isna(b) else np.nan
            rows.append({"zip3":z, "market_type":m, "fc":fc_name, "pmf_alloc":q, "distance_bucket":bkt})

    alloc_df = pd.DataFrame(rows)
    alloc_df.to_csv(OUT_DIR / f"task4_demand_split_{network_name}.csv", index=False)

    agg = alloc_df.groupby(["market_type","distance_bucket"], dropna=False)["pmf_alloc"].sum().reset_index()
    agg = agg.rename(columns={"pmf_alloc":"demand_share"})
    agg.to_csv(OUT_DIR / f"task4d_{network_name}_market_distance_distribution.csv", index=False)
    return agg

def plot_distribution_bars(agg, network_name):
    markets = [m for m in agg["market_type"].dropna().unique().tolist()]
    cats = BUCKET_LABELS
    for m in markets:
        sub = agg[agg["market_type"]==m].copy()
        sub["distance_bucket"] = pd.Categorical(sub["distance_bucket"], categories=cats, ordered=True)
        sub = sub.sort_values("distance_bucket")
        fig, ax = plt.subplots(figsize=(9,4.5))
        ax.bar(sub["distance_bucket"].astype(str), sub["demand_share"])
        ax.set_title(f"Task 4d — {m} demand distribution by distance bucket (90/10) — {network_name}")
        ax.set_xlabel("Distance bucket (miles)")
        ax.set_ylabel("Demand share (PMF alloc)")
        fig.tight_layout()
        out_path = OUT_DIR / f"task4d_{network_name}_{m}_bar.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

# -----------------------------
# Runner
# -----------------------------
def run_for_network(network_name, network_dict):
    assign = load_or_compute_assignment(network_name, network_dict)
    assign4 = build_fulfillment_sets(assign, network_dict)

    # save helper
    out_cols = ["zip3","market_type","pmf_norm","preferred_fc","preferred_fc_zip3",
                "preferred_fc_distance","preferred_bucket","fc_set","fc_count","cluster_id","lat","lon"]
    present_cols = [c for c in out_cols if c in assign4.columns]
    assign4[present_cols].to_csv(OUT_DIR / f"task4_assignment_{network_name}.csv", index=False)

    # (a) clusters
    plot_clusters_map(assign4, network_name)
    # (b) fc count colors
    plot_fc_count_map(assign4, network_name)
    # (c) single-fc-only share
    _ = compute_single_fc_share(assign4, network_name)
    # (d) 90/10 split + plots
    agg = compute_split_distribution(assign4, network_dict, network_name)
    plot_distribution_bars(agg, network_name)

def main():
    for net, fcs in NETWORKS.items():
        run_for_network(net, fcs)

if __name__ == "__main__":
    main()
