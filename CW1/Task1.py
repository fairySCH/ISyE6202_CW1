import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point
import geopandas as gpd
import numpy as np

# Load data
zip3_coordinates = pd.read_csv("zip3_coordinates.csv")  # 'ZIP3', 'Lat', 'Lon'
zip3_pmf = pd.read_csv("zip3_pmf.csv")                  # 'ZIP3', 'PMF'
zip3_market = pd.read_csv("zip3_market.csv")            # 'ZIP3', 'Market', 'State'

# Merge & rename
df = zip3_coordinates.merge(zip3_pmf, on="ZIP3").merge(zip3_market, on="ZIP3")
df = df.rename(columns={"Lat": "latitude", "Lon": "longitude", "PMF": "pmf"})

# Demand assumptions
US_TOTAL_DEMAND = 2_000_000
DOBDA_SHARE = 0.036
DOBDA_SHARE_GROWTH = 0.20
DOBDA_SHARE_ADJUSTED = DOBDA_SHARE * (1 + DOBDA_SHARE_GROWTH)
DOBDA_DEMAND = US_TOTAL_DEMAND * DOBDA_SHARE_ADJUSTED
COMPETITOR_DEMAND = US_TOTAL_DEMAND - DOBDA_DEMAND

# Apply demand to ZIP3
df["dobda_units"] = df["pmf"] * DOBDA_DEMAND
df["competitor_units"] = df["pmf"] * COMPETITOR_DEMAND

# Filter ZIP3s with enough demand
df_vis = df[df["dobda_units"] > 100]

# Plot 3D bar chart
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

x = df_vis["longitude"]
y = df_vis["latitude"]
z = np.zeros(len(df_vis))
dx = dy = 0.3
dz_dobda = df_vis["dobda_units"]
dz_competitor = df_vis["competitor_units"]

ax.bar3d(x, y, z, dx, dy, dz_dobda, color="green", alpha=0.7)
ax.bar3d(x, y, dz_dobda, dx, dy, dz_competitor, color="gray", alpha=0.3)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Unit Demand")
ax.set_title("3D Market Demand by ZIP3: Dobda vs Competitors")
ax.view_init(elev=35, azim=-60)

# Manual legend (text fallback)
ax.text(-125, 24, 1.5e5, "Dobda", color="green", fontsize=12)
ax.text(-125, 22, 1.5e5, "Competitors", color="gray", fontsize=12)




plt.tight_layout()
plt.savefig("dobda_vs_competitors_3d.png", dpi=300)
plt.show()
