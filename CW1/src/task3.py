"""
Dobda Case — Task 3 (a, b, c) Solution Script
Author: ChatGPT
Inputs (expected in CW1/data):
  - zip3_pmf.csv
  - zip3_market.csv  (contains Market column: Primary / Secondary / Tertiary)
  - zip3_coordinates.csv
  - fc_zip3_distance.csv

Outputs (written to CW1/output/task3/):
  - (a) cluster maps: task3a_{network}_map_plotly.html
  - (b) demand share tables and plots:
        task3b_fc_demand_share_{network}.csv
        task3b_fc_market_demand_share_{network}.csv
        task3b_fc_market_demand_share_{network}_plot.png
  - (c) distance distribution tables and plots:
        task3c_market_distance_distribution_{network}.csv
        task3c_fc_market_distance_distribution_{network}.csv
        task3c_market_distance_distribution_{network}_plot.png
  - helper: assignment_{network}.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

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

# Ensure PMF column
if "pmf" not in zip3_pmf.columns:
    zip3_pmf = zip3_pmf.rename(columns={zip3_pmf.columns[-1]: "pmf"})

# Use "Market" column (Primary/Secondary/Tertiary)
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

def plot_clusters_plotly(assign, network_name):
    """Plot clusters using Plotly (interactive HTML)."""
    fc_names = sorted(assign["preferred_fc"].dropna().unique())
    palette = sns.color_palette("husl", n_colors=len(fc_names))
    color_map = {fc: f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
                 for fc, (r,g,b) in zip(fc_names, palette)}

    fig = go.Figure()

    # ZIP dots
    for fc_name, grp in assign.groupby("preferred_fc"):
        fig.add_trace(go.Scattergeo(
            lon=grp["lon"],
            lat=grp["lat"],
            mode="markers",
            name=f"{fc_name} ZIPs",
            marker=dict(size=3, color=color_map.get(fc_name, "gray"), opacity=0.5),
            hoverinfo="text",
            text=[f"ZIP3: {z}, Market: {m}" for z, m in zip(grp["zip3"], grp["market_type"])]
        ))

    # FC triangles
    fc_map = assign.dropna(subset=["preferred_fc_zip3"]) \
                   .groupby("preferred_fc")["preferred_fc_zip3"].first().to_dict()
    for fc_name, fc_zip in fc_map.items():
        row = assign[assign["zip3"] == int(fc_zip)]
        if len(row) > 0:
            fig.add_trace(go.Scattergeo(
                lon=[float(row["lon"].iloc[0])],
                lat=[float(row["lat"].iloc[0])],
                mode="markers+text",
                name=f"{fc_name} (FC)",
                marker=dict(size=12, symbol="triangle-up",
                            color=color_map.get(fc_name, "red"),
                            line=dict(width=1, color="black")),
                text=[fc_name],
                textposition="top center"
            ))

    fig.update_layout(
        title=f"Task 3a — ZIP3 Clusters by Preferred FC ({network_name})",
        geo=dict(
            scope="usa",
            projection=dict(type="albers usa"),
            showland=True, landcolor="rgb(240,240,240)",
            subunitcolor="rgb(100,100,100)",
            countrycolor="rgb(100,100,100)"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5)
    )

    out_path = OUT_DIR / f"task3a_{network_name}_map_plotly.html"
    fig.write_html(out_path)
    print(f"  ✓ Saved interactive map: {out_path}")

def plot_fc_market_demand_share(network_name: str):
    """Grouped bar chart: demand share by FC × Market."""
    csv_path = OUT_DIR / f"task3b_fc_market_demand_share_{network_name}.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df = df.sort_values(["preferred_fc","market_type"])
    plt.figure(figsize=(10,6))
    ax = sns.barplot(data=df, x="preferred_fc", y="demand_share", hue="market_type")

    ymax = df["demand_share"].max() * 1.1
    offset = ymax * 0.01  # 1% of y max as offset

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}",
                        (p.get_x() + p.get_width() / 2., h + offset),
                        ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_ylim(0, ymax)
    plt.title(f"Task 3b — FC × Market Demand Share ({network_name})")
    plt.tight_layout()
    out_path = OUT_DIR / f"task3b_fc_market_demand_share_{network_name}_plot.png"
    plt.savefig(out_path, dpi=220)
    plt.close()

def plot_market_distance_distribution(network_name: str):
    """Grouped bar chart: demand share by Market × Distance bucket."""
    csv_path = OUT_DIR / f"task3c_market_distance_distribution_{network_name}.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    df["distance_bucket"] = pd.Categorical(df["distance_bucket"], categories=BUCKET_LABELS, ordered=True)
    df = df.sort_values(["market_type","distance_bucket"])
    plt.figure(figsize=(12,6))
    ax = sns.barplot(data=df, x="distance_bucket", y="demand_share", hue="market_type")

    ymax = df["demand_share"].max() * 1.1
    offset = ymax * 0.01  # 1% of y max as offset

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x()+p.get_width()/2., h + offset),
                        ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_ylim(0, ymax)
    plt.title(f"Task 3c — Market × Distance Bucket Demand Share ({network_name})")
    plt.tight_layout()
    out_path = OUT_DIR / f"task3c_market_distance_distribution_{network_name}_plot.png"
    plt.savefig(out_path, dpi=220); plt.close()

def compute_task3b(assign, network_name):
    """Compute demand share per FC and per FC×Market."""
    g_fc = assign.groupby("preferred_fc", dropna=False)["pmf_norm"].sum().reset_index()
    g_fc = g_fc.rename(columns={"pmf_norm": "demand_share"})
    g_fc.to_csv(OUT_DIR / f"task3b_fc_demand_share_{network_name}.csv", index=False)

    cats = ["Primary","Secondary","Tertiary"]
    assign["market_type"] = pd.Categorical(assign["market_type"], categories=cats, ordered=True)
    g_fc_mt = assign.groupby(["preferred_fc","market_type"], dropna=False)["pmf_norm"].sum().reset_index()
    g_fc_mt = g_fc_mt.rename(columns={"pmf_norm": "demand_share"}).sort_values(["preferred_fc","market_type"])
    g_fc_mt.to_csv(OUT_DIR / f"task3b_fc_market_demand_share_{network_name}.csv", index=False)

def compute_task3c(assign, network_name):
    """Compute demand distribution by Market × Distance and FC × Market × Distance."""
    sub = assign.dropna(subset=["distance_bucket"]).copy()
    sub["distance_bucket"] = pd.Categorical(sub["distance_bucket"], categories=BUCKET_LABELS, ordered=True)
    mt = sub.groupby(["market_type","distance_bucket"], dropna=False)["pmf_norm"].sum()
    all_mt = sub["market_type"].dropna().unique().tolist()
    full_idx = pd.MultiIndex.from_product([all_mt, BUCKET_LABELS],
                                          names=["market_type","distance_bucket"])
    mt = mt.reindex(full_idx, fill_value=0).reset_index()
    mt = mt.rename(columns={"pmf_norm": "demand_share"})
    mt.to_csv(OUT_DIR / f"task3c_market_distance_distribution_{network_name}.csv", index=False)
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
    plot_clusters_plotly(assign, network_name)
    compute_task3b(assign, network_name)
    compute_task3c(assign, network_name)
    plot_fc_market_demand_share(network_name)
    plot_market_distance_distribution(network_name)

def main():
    for net, fcs in NETWORKS.items():
        run_for_network(net, fcs)

if __name__ == "__main__":
    main()
