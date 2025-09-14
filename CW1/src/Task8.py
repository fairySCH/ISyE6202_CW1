import numpy as np
import pandas as pd
from pathlib import Path

# ========= SINGLE INPUT PATH =========
IN_CSV = Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7/Task7 FC Daily.csv")

# ========= OUTPUT PATH =========
OUT_DIR = Path("/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= PARAMS =========
INIT_INV  = 0.0        # initial DC inventory at the DC
AGG_NET   = False      # True: aggregate across all FCs (network); False: one FC
FC_FILTER = None       # e.g., "FC_01" (only if AGG_NET is False)

# ========= HELPERS =========
def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise RuntimeError(f"Missing any of columns: {candidates}")
    return None

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0)

# ========= LOAD =========
df = pd.read_csv(IN_CSV)

# Support both “pretty” Task7 names and original names
col_date   = pick_col(df, ["Date", "date"])
col_fc     = pick_col(df, ["FC ID", "FC_ID", "fc_id"], required=False)
col_cycle  = pick_col(df, ["FC Average", "fc_average", "fc_cycle"])
col_target = pick_col(df, ["FC Target", "fc_target"])

df[col_date] = pd.to_datetime(df[col_date])

# ========= SCOPE (network vs single FC) =========
if AGG_NET:
    daily = (df.groupby(col_date, as_index=False)[[col_cycle, col_target]]
               .sum().sort_values(col_date).reset_index(drop=True))
else:
    if FC_FILTER and (col_fc in df.columns):
        daily = df[df[col_fc] == FC_FILTER].copy()
        if daily.empty:
            raise RuntimeError(f"No rows for {col_fc}='{FC_FILTER}'.")
        daily = daily.sort_values(col_date).reset_index(drop=True)
    else:
        if col_fc in df.columns:
            first_fc = df[col_fc].iloc[0]
            daily = df[df[col_fc] == first_fc].sort_values(col_date).reset_index(drop=True)
        else:
            daily = df.sort_values(col_date).reset_index(drop=True)

# Standardize columns
daily = daily.rename(columns={col_date: "Date", col_cycle: "Demand", col_target: "Target"})[["Date", "Demand", "Target"]]
daily["Demand"] = to_num(daily["Demand"])
daily["Target"] = to_num(daily["Target"])
daily = daily.sort_values("Date").reset_index(drop=True)
if daily.empty:
    raise RuntimeError("No daily rows to process.")

d = daily["Demand"].to_numpy(float)
t = daily["Target"].to_numpy(float)
n = len(d)

# ========= Strategy A: Follow Demand (Chase) =========
inv_beg = np.zeros(n + 1); inv_beg[0] = INIT_INV
af_prod_chase = np.zeros(n); dc_inv_mid_chase = np.zeros(n)

for i in range(n):
    af_prod_chase[i]    = max(0.0, t[i] - inv_beg[i])
    dc_inv_mid_chase[i] = inv_beg[i] + af_prod_chase[i]
    inv_beg[i+1]        = max(0.0, dc_inv_mid_chase[i] - d[i])

peak_cap_a = float(af_prod_chase.max()) if n else 0.0
peak_mid_a = float(dc_inv_mid_chase.max()) if n else 0.0
peak_eod_a = float(inv_beg[1:].max()) if n else 0.0

# ========= Strategy B: Steady Rate (Level) =========
cum_d_prev = np.concatenate(([0.0], np.cumsum(d)[:-1]))
rate_cands = (t - INIT_INV + cum_d_prev) / (np.arange(n) + 1)
rate = float(max(0.0, rate_cands.max())) if n else 0.0

inv_beg_b = np.zeros(n + 1); inv_beg_b[0] = INIT_INV
dc_inv_mid_level = np.zeros(n)

for i in range(n):
    dc_inv_mid_level[i] = inv_beg_b[i] + rate
    inv_beg_b[i+1]      = max(0.0, dc_inv_mid_level[i] - d[i])

peak_cap_b = rate
peak_mid_b = float(dc_inv_mid_level.max()) if n else 0.0
peak_eod_b = float(inv_beg_b[1:].max()) if n else 0.0

# ========= Outputs (your requested names) =========
df_a = pd.DataFrame({
    "Date": daily["Date"],
    "Demand": d,
    "Target": t,
    "AF Production": af_prod_chase,
    "DC Inventory": dc_inv_mid_chase,
    "DC Inventory EOD": inv_beg[1:],
})

df_b = pd.DataFrame({
    "Date": daily["Date"],
    "Demand": d,
    "Target": t,
    "AF Steady Rate": rate,
    "DC Inventory": dc_inv_mid_level,
    "DC Inventory EOD": inv_beg_b[1:],
})

df_sum = pd.DataFrame({
    "Peak Loads":   ["af_peak_daily_cap", "dc_peak_mid", "dc_peak_eod"],
    "Follow Demand":[peak_cap_a,            peak_mid_a,    peak_eod_a],
    "Steady Rate":  [peak_cap_b,            peak_mid_b,    peak_eod_b],
})

# File names (keeping your “Startegy” spelling)
p_a = OUT_DIR / "Task8 Strategy A.csv"
p_b = OUT_DIR / "Task8 Startegy B.csv"
p_s = OUT_DIR / "Task8 AF Startegy Summary.csv"

df_a.to_csv(p_a, index=False)
df_b.to_csv(p_b, index=False)
df_sum.to_csv(p_s, index=False)

# Combined Excel
with pd.ExcelWriter(OUT_DIR / "Task8 Report.xlsx") as xw:
    df_a.to_excel(xw, sheet_name="Strategy A", index=False)
    df_b.to_excel(xw, sheet_name="Strategy B", index=False)
    df_sum.to_excel(xw, sheet_name="Summary", index=False)

print("✅ Wrote:")
print(" -", p_a)
print(" -", p_b)
print(" -", p_s)
print(" -", OUT_DIR / "Task8 Report.xlsx")
