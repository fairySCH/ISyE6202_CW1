"""
Task: Demand Analysis + Scenario Plots + Stochastic Simulations + ZIP/Market/State breakdowns
Author: ChatGPT (Full Version)

Layout (fixed paths for GitHub project):
- project_root/
    - README.md
    - LICENSE
    - CW1/
        - data/
            demand_seasonalities.csv
            fc_zip3_distance.csv
            msa.csv
            zip3_coordinates.csv
            zip3_market.csv
            zip3_pmf.csv
        - output/   <-- all outputs will be written here (HTML/PNG/CSV)
        - src/
            task_demand_analysis.py  <-- this file

What this script does:
1) Loads inputs from CW1/data (zip3 PMF, seasonality, coordinates, market labels, etc.)
2) Defines demand growth scenarios (optimistic/expected/conservative × min/median/max)
3) Builds a 364-day calendar, distributes totals daily, and plots:
   - Daily demand across 9 scenarios
   - Cumulative demand bands (min..max) + expected median
4) Creates ZIP3-level bar charts (Top 20/100 by demand), and Market/State bar charts
5) Runs stochastic daily paths using uncertain seasonality (with quantile bands)
6) ZIP-level lines with uncertainty bands and “spaghetti” samples
7) Spatial (ZIP3) distribution with error bars of annual demand (per scenario)
8) Writes all plots as HTML (and PNG if Kaleido available) + saves useful CSVs
"""

from pathlib import Path
import os
import locale
from math import ceil

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------
# Helpers: safe PNG export (fallback to HTML only if Kaleido not present)
# ---------------------------------------------------------------------
def _safe_write_image(fig, path_png: Path):
    """Try writing PNG via Plotly/Kaleido. If not available, skip silently."""
    try:
        fig.write_image(str(path_png))
    except Exception as e:
        print(f"[warn] PNG export skipped for {path_png.name} (install kaleido to enable). Reason: {e}")


def _slug(s: str, maxlen: int = 80) -> str:
    """Build a safe filename stub from a title string."""
    return (
        (s or "plot")
        .lower()
        .replace(" ", "_")
        .replace("—", "_")
        .replace("–", "_")
        .replace("/", "_")
    )[:maxlen]


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------
    # Path setup (fixed to CW1/*)
    # -----------------------------
    BASE_DIR = Path(__file__).resolve().parents[1]   # project_root/CW1/
    DATA_DIR = BASE_DIR / "data"
    OUT_DIR = BASE_DIR / "output/task1_2"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting demand analysis...")
    print("BASE_DIR =", BASE_DIR)
    print("DATA_DIR =", DATA_DIR)
    print("OUT_DIR  =", OUT_DIR)

    # -----------------------------
    # Locale (for date formatting)
    # -----------------------------
    try:
        current_locale = locale.getlocale(locale.LC_TIME)
        locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    except Exception:
        pass  # Not critical for plotting

    # -----------------------------
    # File paths (fixed)
    # -----------------------------
    paths = {
        "seasonalities": DATA_DIR / "demand_seasonalities.csv",
        "fc_dist": DATA_DIR / "fc_zip3_distance.csv",
        "msa": DATA_DIR / "msa.csv",
        "zip3_coords": DATA_DIR / "zip3_coordinates.csv",
        "zip3_market": DATA_DIR / "zip3_market.csv",
        "zip3_pmf": DATA_DIR / "zip3_pmf.csv",
    }

    # -----------------------------
    # Utility functions
    # -----------------------------
    def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    def ensure_zip3_str(series: pd.Series) -> pd.Series:
        # keep 3-digit strings, preserving leading zeros
        return series.astype(str).str.extract(r'(\d{1,3})', expand=False).str.zfill(3)

    # -----------------------------
    # Load datasets
    # -----------------------------
    df_seasonalities = standardize_cols(pd.read_csv(paths["seasonalities"]))

    df_fcdist = standardize_cols(pd.read_csv(paths["fc_dist"]))
    # Ensure zip3 column is 3-digit text
    if "zip3" in df_fcdist.columns:
        df_fcdist["zip3"] = ensure_zip3_str(df_fcdist["zip3"])

    df_msa = standardize_cols(pd.read_csv(paths["msa"]))
    if "3-digit_zip_code" in df_msa.columns:
        df_msa["3-digit_zip_code"] = ensure_zip3_str(df_msa["3-digit_zip_code"])

    df_zip3coords = standardize_cols(pd.read_csv(paths["zip3_coords"]))
    df_zip3coords["zip3"] = ensure_zip3_str(df_zip3coords["zip3"])

    df_zip3market = standardize_cols(pd.read_csv(paths["zip3_market"]))
    df_zip3market["zip3"] = ensure_zip3_str(df_zip3market["zip3"])

    df_zip3pmf = standardize_cols(pd.read_csv(paths["zip3_pmf"]))
    df_zip3pmf["zip3"] = ensure_zip3_str(df_zip3pmf["zip3"])
    df_zip3pmf["pmf"] = pd.to_numeric(df_zip3pmf["pmf"], errors="coerce").fillna(0.0)
    # Normalize PMF to sum 1
    s_pmf = df_zip3pmf["pmf"].sum()
    if s_pmf <= 0:
        raise ValueError("zip3_pmf.csv has non-positive total PMF.")
    df_zip3pmf["pmf"] = df_zip3pmf["pmf"] / s_pmf

    # -----------------------------
    # Parameters and basic timeline
    # -----------------------------
    # Growth assumptions
    overall_market_grow = 0.075
    overall_market_grow_conservative = 0.04
    overall_market_grow_optimistic = 0.12

    # Dobda market share growth
    minimum_market_share_grow_yearly = 0.15
    median_market_share_grow_yearly = 0.20
    maximum_market_share_grow_yearly = 0.25
    current_market_share = 0.036

    # General context
    average_price = 3000  # dollars (not used in plots but kept for completeness)
    total_units_overall = 2_000_000

    # Daily grid: use 364 days = 52 weeks × 7 days
    N = 364
    weeks = np.tile(np.arange(1, 53), ceil(N / 52))[:N]
    days = np.tile(np.arange(1, 8), ceil(N / 7))[:N]
    start_date = pd.Timestamp("2026-01-01")

    df_global = pd.DataFrame({
        "semaine": weeks,
        "jour_semaine": days,
        "date": start_date + pd.to_timedelta(np.arange(N), unit="D")
    })

    # -----------------------------
    # Demand growth arithmetic
    # -----------------------------
    # 2025 baseline (approx): overall * current share
    units_dodba_current = total_units_overall * current_market_share

    # 2026 overall volumes
    overall_units_2026 = total_units_overall * (1 + overall_market_grow)
    overall_units_2026_optimistic = total_units_overall * (1 + overall_market_grow_optimistic)
    overall_units_2026_conservative = total_units_overall * (1 + overall_market_grow_conservative)

    # 2026 Dobda shares (median/min/max growth)
    market_share_dodba_2026 = current_market_share * (1 + median_market_share_grow_yearly)
    market_share_dodba_2026_min = current_market_share * (1 + minimum_market_share_grow_yearly)
    market_share_dodba_2026_max = current_market_share * (1 + maximum_market_share_grow_yearly)

    # 2026 Dobda units under overall scenarios
    units_2026 = overall_units_2026 * market_share_dodba_2026
    units_2026_optimistic = overall_units_2026_optimistic * market_share_dodba_2026
    units_2026_conservative = overall_units_2026_conservative * market_share_dodba_2026

    units_2026_min = overall_units_2026 * market_share_dodba_2026_min
    units_2026_optimistic_min = overall_units_2026_optimistic * market_share_dodba_2026_min
    units_2026_conservative_min = overall_units_2026_conservative * market_share_dodba_2026_min

    units_2026_max = overall_units_2026 * market_share_dodba_2026_max
    units_2026_optimistic_max = overall_units_2026_optimistic * market_share_dodba_2026_max
    units_2026_conservative_max = overall_units_2026_conservative * market_share_dodba_2026_max

    # -----------------------------
    # Daily demand columns (9 scenarios)
    # -----------------------------
    daily_cols = {
        "expected_min":        units_2026_min / N,
        "expected_median":     units_2026 / N,
        "expected_max":        units_2026_max / N,
        "optimistic_min":      units_2026_optimistic_min / N,
        "optimistic_median":   units_2026_optimistic / N,
        "optimistic_max":      units_2026_optimistic_max / N,
        "conservative_min":    units_2026_conservative_min / N,
        "conservative_median": units_2026_conservative / N,
        "conservative_max":    units_2026_conservative_max / N,
    }
    for k, v in daily_cols.items():
        df_global[f'units_daily__{k}'] = float(v)

    # -----------------------------------------------------------------
    # Plot 0: Overview of total market vs DoDBA (2025/2026, 3 scenarios)
    # -----------------------------------------------------------------
    scenarios = {
        "optimist": {"2025": total_units_overall, "2026": overall_units_2026_optimistic},
        "expected": {"2025": total_units_overall, "2026": overall_units_2026},
        "conservative": {"2025": total_units_overall, "2026": overall_units_2026_conservative},
    }
    scenario_dodba = {
        "optimist": {"2025": units_dodba_current, "2026": units_2026_optimistic},
        "expected": {"2025": units_dodba_current, "2026": units_2026},
        "conservative": {"2025": units_dodba_current, "2026": units_2026_conservative},
    }
    sc_order = ["optimist", "expected", "conservative"]
    years = ["2025", "2026"]

    x_scenarios, x_years = [], []
    y_market, y_dodba = [], []
    for s in sc_order:
        for y in years:
            x_scenarios.append(s)
            x_years.append(y)
            y_market.append(float(scenarios[s][y]))
            y_dodba.append(float(scenario_dodba[s][y]))

    fig_over = go.Figure()
    fig_over.add_trace(go.Bar(
        x=[x_scenarios, x_years],
        y=y_market,
        name="Overall Demand",
        opacity=0.45,
        text=[f"{v:,}" for v in y_market],
        hovertemplate="<b>%{x[0]}</b> — %{x[1]}<br>Market: %{y:,}<extra></extra>",
    ))
    fig_over.add_trace(go.Bar(
        x=[x_scenarios, x_years],
        y=y_dodba,
        name="DoDBA",
        opacity=0.95,
        text=[f"{int(v):,}" for v in y_dodba],
        hovertemplate="<b>%{x[0]}</b> — %{x[1]}<br>DoDBA: %{y:,}<extra></extra>",
    ))
    fig_over.update_layout(
        title="Overview of Overall Market vs DoDBA (2025 → 2026)",
        barmode="overlay",
        xaxis_title="Scenario / Year",
        yaxis_title="Units",
        yaxis_tickformat=",",
        bargap=0.25,
    )
    name_over = _slug(fig_over.layout.title.text)
    fig_over.write_html(OUT_DIR / f"{name_over}.html")
    _safe_write_image(fig_over, OUT_DIR / f"{name_over}.png")

    # -----------------------------------------------------------------
    # Plot 1: Daily demand across 9 scenarios
    # -----------------------------------------------------------------
    cols_9 = [f'units_daily__{k}' for k in daily_cols.keys()]
    df_plot = (
        df_global[['date'] + cols_9]
        .rename(columns={f'units_daily__{k}': k for k in daily_cols.keys()})
        .melt('date', var_name='scenario', value_name='units')
    )

    fig1 = px.line(
        df_plot, x='date', y='units', color='scenario',
        title="Daily demand – 9 scenarios (min/median/max × optimistic/expected/conservative)"
    )
    name_fig1 = _slug(fig1.layout.title.text)
    fig1.write_html(OUT_DIR / f"{name_fig1}.html")
    _safe_write_image(fig1, OUT_DIR / f"{name_fig1}.png")

    # -----------------------------------------------------------------
    # Plot 2: Cumulative demand band (conservative_min .. optimistic_max)
    # -----------------------------------------------------------------
    df_plot = df_plot.sort_values(['scenario', 'date'])
    df_plot['units_cum'] = df_plot.groupby('scenario', sort=False)['units'].cumsum()

    band_cum = (
        df_plot.query("scenario in ['conservative_min', 'optimistic_max']")
               .pivot(index='date', columns='scenario', values='units_cum')
               .sort_index()
    )

    fig_band = go.Figure()
    fig_band.add_trace(go.Scatter(
        x=band_cum.index, y=band_cum['conservative_min'],
        name="Conservative min (cum.)",
        line=dict(width=2),
        hovertemplate="Date: %{x}<br>Conservative min (cum.): %{y:,.2f}<extra></extra>"
    ))
    fig_band.add_trace(go.Scatter(
        x=band_cum.index, y=band_cum['optimistic_max'],
        name="Optimistic max (cum.)",
        fill="tonexty",
        fillcolor="rgba(66,133,244,0.25)",
        line=dict(width=2),
        hovertemplate="Date: %{x}<br>Optimistic max (cum.): %{y:,.2f}<extra></extra>"
    ))
    # Optional: median cumulative for context
    if 'expected_median' in df_plot['scenario'].unique():
        med = (df_plot[df_plot['scenario'] == 'expected_median'].sort_values('date'))
        fig_band.add_trace(go.Scatter(
            x=med['date'], y=med['units_cum'],
            name="Expected median (cum.)",
            line=dict(width=2, dash="dash"),
            hovertemplate="Date: %{x}<br>Expected median (cum.): %{y:,.2f}<extra></extra>"
        ))
    fig_band.update_layout(
        title="Cumulative Demand — band [conservative_min, optimistic_max]",
        xaxis_title="Date",
        yaxis_title="Cumulative units",
        yaxis_tickformat=",",
        hovermode="x unified",
    )
    name_band = _slug(fig_band.layout.title.text)
    fig_band.write_html(OUT_DIR / f"{name_band}.html")
    _safe_write_image(fig_band, OUT_DIR / f"{name_band}.png")

    # -----------------------------------------------------------------
    # CSV: Save main global frame + normalized zip3 PMF
    # -----------------------------------------------------------------
    df_global.to_csv(OUT_DIR / "df_global.csv", index=False)
    df_zip3pmf.to_csv(OUT_DIR / "df_zip3pmf_normalized.csv", index=False)

    # -----------------------------------------------------------------
    # Seasonality-based (week/day) daily PMF and stochastic totals
    # -----------------------------------------------------------------
    # Extract required columns for seasonality
    df_season_week = df_seasonalities[["week_of_year", "proportion"]].copy()
    df_day_week = df_seasonalities[["day_of_week", "proportion.1"]].iloc[:7].copy()

    # Convert with tolerant numeric parsing
    def to_prop(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        s = (
            s.str.replace('%', '', regex=False)
             .str.replace('\u202f', '')
             .str.replace('\xa0', '')
             .str.replace(',', '.', regex=False)
        )
        vals = pd.to_numeric(s, errors='coerce')
        # If any had %, they’re already stripped above; treat as fraction if you prefer:
        # Here we assume "proportion" columns are in percent. Convert to [0,1].
        return vals / 100.0

    pmf_w = to_prop(df_season_week['proportion']).to_numpy()
    pmf_d = to_prop(df_day_week['proportion.1']).to_numpy()

    # Random jitter range (20% for weeks, 15% for days)
    rng = np.random.default_rng(2027)
    eps_w, eps_d = 0.20, 0.15
    sample_w = rng.uniform(pmf_w * (1 - eps_w), pmf_w * (1 + eps_w))
    sample_d = rng.uniform(pmf_d * (1 - eps_d), pmf_d * (1 + eps_d))

    pmf_w = sample_w / sample_w.sum()
    pmf_d = sample_d / sample_d.sum()

    # Build daily PMF for the calendar (52×7 → 364)
    w_vec = np.take(pmf_w, df_global['semaine'].astype(int).to_numpy() - 1)
    d_vec = np.take(pmf_d, df_global['jour_semaine'].astype(int).to_numpy() - 1)
    pmf_raw = w_vec * d_vec
    df_global2 = df_global[["semaine", "jour_semaine", "date"]].copy()
    df_global2['pmf_daily'] = pmf_raw / pmf_raw.sum()

    # Uncertainty on total 2026 demand (base std = 5% of expected)
    standard_deviation = overall_units_2026 * market_share_dodba_2026 * 0.05

    # Totals with normal noise (68%, 95%, 99%)
    units_2026_expected_distnormal_68 = rng.normal(units_2026, standard_deviation)
    units_2026_optimistic_distnormal_68 = rng.normal(units_2026_optimistic, standard_deviation)
    units_2026_conservative_distnormal_68 = rng.normal(units_2026_conservative, standard_deviation)

    units_2026_expected_distnormal_95 = rng.normal(units_2026, standard_deviation * 2)
    units_2026_optimistic_distnormal_95 = rng.normal(units_2026_optimistic, standard_deviation * 2)
    units_2026_conservative_distnormal_95 = rng.normal(units_2026_conservative, standard_deviation * 2)

    units_2026_expected_distnormal_99 = rng.normal(units_2026, standard_deviation * 2.5)
    units_2026_optimistic_distnormal_99 = rng.normal(units_2026_optimistic, standard_deviation * 2.5)
    units_2026_conservative_distnormal_99 = rng.normal(units_2026_conservative, standard_deviation * 2.5)

    df_global2['units_2026_expected_distnormal_68'] = units_2026_expected_distnormal_68 * df_global2['pmf_daily']
    df_global2['units_2026_optimistic_distnormal_68'] = units_2026_optimistic_distnormal_68 * df_global2['pmf_daily']
    df_global2['units_2026_conservative_distnormal_68'] = units_2026_conservative_distnormal_68 * df_global2['pmf_daily']

    df_global2['units_2026_expected_distnormal_95'] = units_2026_expected_distnormal_95 * df_global2['pmf_daily']
    df_global2['units_2026_optimistic_distnormal_95'] = units_2026_optimistic_distnormal_95 * df_global2['pmf_daily']
    df_global2['units_2026_conservative_distnormal_95'] = units_2026_conservative_distnormal_95 * df_global2['pmf_daily']

    df_global2['units_2026_expected_distnormal_99'] = units_2026_expected_distnormal_99 * df_global2['pmf_daily']
    # support for potential double-underscore typo in original code
    df_global2['units_2026_optimistic_distnormal_99'] = units_2026_optimistic_distnormal_99 * df_global2['pmf_daily']
    df_global2['units_2026_conservative_distnormal_99'] = units_2026_conservative_distnormal_99 * df_global2['pmf_daily']

    # -----------------------------
    # Dropdown plots for (68% / 95% / 99%) daily & cumulative
    # -----------------------------
    cols = set(df_global2.columns)

    def pick_first_present(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    levels = {
        "68%": [
            ("expected",     pick_first_present(["units_2026_expected_distnormal_68"])),
            ("optimistic",   pick_first_present(["units_2026_optimistic_distnormal_68"])),
            ("conservative", pick_first_present(["units_2026_conservative_distnormal_68"])),
        ],
        "95%": [
            ("expected",     pick_first_present(["units_2026_expected_distnormal_95"])),
            ("optimistic",   pick_first_present(["units_2026_optimistic_distnormal_95"])),
            ("conservative", pick_first_present(["units_2026_conservative_distnormal_95"])),
        ],
        "99%": [
            ("expected",     pick_first_present(["units_2026_expected_distnormal_99"])),
            ("optimistic",   pick_first_present(["units_2026_optimistic_distnormal_99"])),
            ("conservative", pick_first_present(["units_2026_conservative_distnormal_99"])),
        ],
    }
    levels = {lvl: [(lab, col) for (lab, col) in triplet if col is not None]
              for lvl, triplet in levels.items()}
    levels = {lvl: triplet for lvl, triplet in levels.items() if len(triplet) > 0}
    level_names = list(levels.keys())
    if not level_names:
        raise ValueError("No expected distribution columns found in df_global2.")

    # Daily dropdown
    fig_daily = go.Figure()
    trace_count_per_level = []
    for li, lvl in enumerate(level_names):
        triplet = levels[lvl]
        trace_count_per_level.append(len(triplet))
        for lab, col in triplet:
            fig_daily.add_trace(go.Scatter(
                x=df_global2['date'], y=df_global2[col],
                mode='lines', name=f"{lab} ({lvl})",
                visible=(li == 0),
                hovertemplate="Date=%{x}<br>Units=%{y:.2f}<extra>" + lab + " " + lvl + "</extra>"
            ))
    buttons = []
    total_traces = sum(trace_count_per_level)
    starts = np.cumsum([0] + trace_count_per_level[:-1])
    for li, lvl in enumerate(level_names):
        vis = [False] * total_traces
        start = int(starts[li]); count = int(trace_count_per_level[li])
        vis[start:start + count] = [True] * count
        buttons.append(dict(
            label=lvl,
            method="update",
            args=[{"visible": vis},
                  {"title": f"Daily demand — scenarios (optimistic/expected/conservative) — level: {lvl}",
                   "yaxis": {"title": "Units"},
                   "xaxis": {"title": "Date"}}]
        ))
    fig_daily.update_layout(
        title=f"Daily demand — scenarios (optimistic/expected/conservative) — level: {level_names[0]}",
        xaxis_title="Date",
        yaxis_title="Units",
        updatemenus=[dict(type="dropdown", direction="down",
                          x=1.02, xanchor="left", y=1, buttons=buttons)]
    )
    name_daily = _slug(fig_daily.layout.title.text)
    fig_daily.write_html(OUT_DIR / f"{name_daily}.html")
    _safe_write_image(fig_daily, OUT_DIR / f"{name_daily}.png")

    # Cumulative dropdown
    fig_cum = go.Figure()
    for li, lvl in enumerate(level_names):
        triplet = levels[lvl]
        for lab, col in triplet:
            fig_cum.add_trace(go.Scatter(
                x=df_global2['date'], y=df_global2[col].cumsum(),
                mode='lines', name=f"{lab} ({lvl})",
                visible=(li == 0),
                hovertemplate="Date=%{x}<br>Cumulative=%{y:.2f}<extra>" + lab + " " + lvl + "</extra>"
            ))
    buttons_cum = []
    for li, lvl in enumerate(level_names):
        vis = [False] * total_traces
        start = int(starts[li]); count = int(trace_count_per_level[li])
        vis[start:start + count] = [True] * count
        buttons_cum.append(dict(
            label=lvl,
            method="update",
            args=[{"visible": vis},
                  {"title": f"Cumulative demand — scenarios (optimistic/expected/conservative) — level: {lvl}",
                   "yaxis": {"title": "Cumulative units"},
                   "xaxis": {"title": "Date"}}]
        ))
    fig_cum.update_layout(
        title=f"Cumulative demand — scenarios (optimistic/expected/conservative) — level: {level_names[0]}",
        xaxis_title="Date",
        yaxis_title="Cumulative units",
        updatemenus=[dict(type="dropdown", direction="down",
                          x=1.02, xanchor="left", y=1, buttons=buttons_cum)]
    )
    name_cum = _slug(fig_cum.layout.title.text)
    fig_cum.write_html(OUT_DIR / f"{name_cum}.html")
    _safe_write_image(fig_cum, OUT_DIR / f"{name_cum}.png")

    # -----------------------------------------------------------------
    # Scenario selector plots (spaghetti + 5–95% bands) using path simulation
    # -----------------------------------------------------------------
    # Build simulation utilities
    def get_base_pmfs(df_season_week: pd.DataFrame, df_day_week: pd.DataFrame):
        wk = df_season_week.sort_values('week_of_year').rename(
            columns={'week_of_year': 'semaine', 'proportion': 'pmf'}).copy()
        dy = df_day_week.sort_values('day_of_week').rename(
            columns={'day_of_week': 'jour_semaine', 'proportion.1': 'pmf'}).copy()
        wk['pmf'] = to_prop(wk['pmf']); dy['pmf'] = to_prop(dy['pmf'])
        pmf_w = wk.set_index('semaine')['pmf'].reindex(range(1, 53), fill_value=0.0).to_numpy()
        pmf_d = dy.set_index('jour_semaine')['pmf'].reindex(range(1, 8), fill_value=0.0).to_numpy()
        pmf_w = pmf_w / pmf_w.sum(); pmf_d = pmf_d / pmf_d.sum()
        return pmf_w, pmf_d

    def sample_pmfs_uniform(pmf_w_base, pmf_d_base, eps_w=0.20, eps_d=0.15, n_paths=200, rng=None):
        if rng is None:
            rng = np.random.default_rng(2027)
        w = rng.uniform(pmf_w_base * (1 - eps_w), pmf_w_base * (1 + eps_w), size=(n_paths, pmf_w_base.size))
        d = rng.uniform(pmf_d_base * (1 - eps_d), pmf_d_base * (1 + eps_d), size=(n_paths, pmf_d_base.size))
        w /= w.sum(axis=1, keepdims=True)
        d /= d.sum(axis=1, keepdims=True)
        return w, d

    def make_daily_pmfs_for_paths(pmf_w_paths, pmf_d_paths, week_idx, day_idx):
        w_mat = pmf_w_paths[:, week_idx]
        d_mat = pmf_d_paths[:, day_idx]
        pmf_raw = w_mat * d_mat
        pmf_daily = pmf_raw / pmf_raw.sum(axis=1, keepdims=True)
        return pmf_daily

    def sample_totals(n_paths, loc, std=None, model="normal", band=0.10, rng=None):
        if rng is None:
            rng = np.random.default_rng(2027)
        if model == "normal":
            if std is None:
                raise ValueError("std required for model='normal'")
            return rng.normal(loc, std, size=n_paths)
        elif model == "uniform":
            return rng.uniform(loc * (1 - band), loc * (1 + band), size=n_paths)
        else:
            raise ValueError("model must be 'normal' or 'uniform'")

    def simulate_paths(
        df_global2, df_season_week, df_day_week,
        n_paths=200, eps_w=0.20, eps_d=0.15,
        total_model="normal", sigma_mult=1.0, uniform_band=0.10,
        seed=2027
    ):
        rng = np.random.default_rng(seed)
        dates = pd.to_datetime(df_global2['date']).to_numpy()
        week_idx = df_global2['semaine'].astype(int).to_numpy() - 1
        day_idx = df_global2['jour_semaine'].astype(int).to_numpy() - 1

        pmf_w_base, pmf_d_base = get_base_pmfs(df_season_week, df_day_week)
        pmf_w_paths, pmf_d_paths = sample_pmfs_uniform(pmf_w_base, pmf_d_base,
                                                       eps_w=eps_w, eps_d=eps_d,
                                                       n_paths=n_paths, rng=rng)
        pmf_daily_paths = make_daily_pmfs_for_paths(pmf_w_paths, pmf_d_paths, week_idx, day_idx)

        scenarios_ = {
            "expected":     units_2026,
            "optimistic":   units_2026_optimistic,
            "conservative": units_2026_conservative,
        }
        out = {}
        for scen, loc in scenarios_.items():
            if total_model == "normal":
                std = sigma_mult * float(standard_deviation)
                totals = sample_totals(n_paths, loc=loc, std=std, model="normal", rng=rng)
            else:
                totals = sample_totals(n_paths, loc=loc, model="uniform", band=uniform_band, rng=rng)

            paths_daily = pmf_daily_paths * totals[:, None]
            paths_cum = paths_daily.cumsum(axis=1)

            q5 = np.quantile(paths_daily, 0.05, axis=0)
            q50 = np.quantile(paths_daily, 0.50, axis=0)
            q95 = np.quantile(paths_daily, 0.95, axis=0)
            q5_c = np.quantile(paths_cum, 0.05, axis=0)
            q50_c = np.quantile(paths_cum, 0.50, axis=0)
            q95_c = np.quantile(paths_cum, 0.95, axis=0)

            out[scen] = {
                "dates": dates,
                "paths_daily": paths_daily,
                "paths_cum": paths_cum,
                "q5": q5, "q50": q50, "q95": q95,
                "q5_cum": q5_c, "q50_cum": q50_c, "q95_cum": q95_c
            }
        return out

    sim = simulate_paths(
        df_global2, df_season_week, df_day_week,
        n_paths=200,
        eps_w=0.20, eps_d=0.15,
        total_model="normal",
        sigma_mult=3.0,      # ≈ 99% band
        uniform_band=0.10,
        seed=2027
    )

    def plot_scenario_switch(sim, cumulative=False, show_samples=12, title_prefix="Scenario demand (uncertain PMF)"):
        scenarios_ = [s for s in ["optimistic", "expected", "conservative"] if s in sim]
        colors = {
            "expected":    ("rgba(31,119,180,0.20)", "rgb(31,119,180)"),
            "optimistic":  ("rgba(44,160,44,0.20)",  "rgb(44,160,44)"),
            "conservative":("rgba(214,39,40,0.20)",  "rgb(214,39,40)")
        }
        fig = go.Figure()
        traces_per_scenario, total_traces = [], 0
        for scen in scenarios_:
            pack = sim[scen]
            x = pack["dates"]
            low  = pack["q5_cum"] if cumulative else pack["q5"]
            med  = pack["q50_cum"] if cumulative else pack["q50"]
            high = pack["q95_cum"] if cumulative else pack["q95"]
            y_paths = pack["paths_cum"] if cumulative else pack["paths_daily"]
            band_color, line_color = colors[scen]
            start_idx = total_traces

            # Band
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([high, low[::-1]]),
                fill="toself", line=dict(width=0),
                fillcolor=band_color, name=f"{scen} (5–95%)",
                hoverinfo="skip", visible=(scen == scenarios_[0])
            ))
            # Median
            fig.add_trace(go.Scatter(
                x=x, y=med, mode="lines",
                line=dict(color=line_color, width=2),
                name=f"{scen} median", visible=(scen == scenarios_[0])
            ))
            # Optional spaghetti
            k = min(show_samples, y_paths.shape[0])
            for i in range(k):
                fig.add_trace(go.Scatter(
                    x=x, y=y_paths[i], mode="lines",
                    line=dict(color=line_color, width=1),
                    opacity=0.25, showlegend=False,
                    visible=(scen == scenarios_[0])
                ))
            count = 2 + k
            traces_per_scenario.append((start_idx, count))
            total_traces += count

        # Dropdown to switch scenarios
        buttons = []
        for scen_idx, scen in enumerate(scenarios_):
            vis = [False] * total_traces
            start, count = traces_per_scenario[scen_idx]
            for j in range(start, start + count):
                vis[j] = True
            buttons.append(dict(
                label=scen,
                method="update",
                args=[{"visible": vis},
                      {"title": f"{title_prefix} — {scen} — {'cumulative' if cumulative else 'daily'}",
                       "yaxis": {"title": "Units"},
                       "xaxis": {"title": "Date"}}]
            ))
        fig.update_layout(
            title=f"{title_prefix} — {scenarios_[0]} — {'cumulative' if cumulative else 'daily'}",
            xaxis_title="Date", yaxis_title="Units",
            updatemenus=[dict(type="dropdown", direction="down", x=1.02, xanchor="left", y=1, buttons=buttons)]
        )
        return fig

    fig_switch_daily = plot_scenario_switch(sim, cumulative=False, show_samples=12,
                                            title_prefix="Scenario demand (daily, uncertain PMF)")
    name_switch_daily = _slug(fig_switch_daily.layout.title.text)
    fig_switch_daily.write_html(OUT_DIR / f"{name_switch_daily}.html")
    _safe_write_image(fig_switch_daily, OUT_DIR / f"{name_switch_daily}.png")

    fig_switch_cum = plot_scenario_switch(sim, cumulative=True, show_samples=12,
                                          title_prefix="Scenario demand (cumulative, uncertain PMF)")
    name_switch_cum = _slug(fig_switch_cum.layout.title.text)
    fig_switch_cum.write_html(OUT_DIR / f"{name_switch_cum}.html")
    _safe_write_image(fig_switch_cum, OUT_DIR / f"{name_switch_cum}.png")

    # -----------------------------------------------------------------
    # ZIP-level helpers and plots
    # -----------------------------------------------------------------
    def _get_zip_p(zip3):
        z = str(zip3).zfill(3)
        tmp = df_zip3pmf.copy()
        tmp['zip3'] = tmp['zip3'].astype(str).str.zfill(3)
        # ensure normalized
        if not np.isclose(tmp['pmf'].sum(), 1.0):
            tmp['pmf'] = tmp['pmf'] / tmp['pmf'].sum()
        row = tmp.loc[tmp['zip3'] == z]
        if row.empty:
            raise ValueError(f"ZIP3 {z} not found in df_zip3pmf.")
        return z, float(row['pmf'].iloc[0])

    def plot_zip_with_sim(sim, zip3, cumulative=False, show_samples=10,
                          title_prefix="ZIP demand (with uncertainty)"):
        z, p = _get_zip_p(zip3)
        scenarios_ = [s for s in ["optimistic", "expected", "conservative"] if s in sim]
        colors = {
            "expected":    ("rgba(31,119,180,0.20)", "rgb(31,119,180)"),
            "optimistic":  ("rgba(44,160,44,0.20)",  "rgb(44,160,44)"),
            "conservative":("rgba(214,39,40,0.20)",  "rgb(214,39,40)")
        }
        fig = go.Figure()
        for scen in scenarios_:
            pack = sim[scen]
            x = pack["dates"]
            low  = (pack["q5_cum"]  if cumulative else pack["q5"])  * p
            med  = (pack["q50_cum"] if cumulative else pack["q50"]) * p
            high = (pack["q95_cum"] if cumulative else pack["q95"]) * p
            y_paths = (pack["paths_cum"] if cumulative else pack["paths_daily"]) * p
            band_color, line_color = colors[scen]

            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([high, low[::-1]]),
                fill="toself", line=dict(width=0),
                fillcolor=band_color, name=f"{scen} (5–95%)", hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=x, y=med, mode="lines",
                line=dict(color=line_color, width=2),
                name=f"{scen} median"
            ))
            k = min(show_samples, y_paths.shape[0])
            for i in range(k):
                fig.add_trace(go.Scatter(
                    x=x, y=y_paths[i], mode="lines",
                    line=dict(color=line_color, width=1),
                    opacity=0.25, showlegend=False
                ))
        fig.update_layout(
            title=f"{title_prefix} — ZIP3 {z} — {'cumulative' if cumulative else 'daily'}",
            xaxis_title="Date", yaxis_title="Units"
        )
        return fig

    # Example ZIP: 850 daily + cumulative
    fig_zip850_d = plot_zip_with_sim(sim, zip3='850', cumulative=False, show_samples=10)
    name_zip850_d = _slug(fig_zip850_d.layout.title.text)
    fig_zip850_d.write_html(OUT_DIR / f"{name_zip850_d}.html")
    _safe_write_image(fig_zip850_d, OUT_DIR / f"{name_zip850_d}.png")

    fig_zip850_c = plot_zip_with_sim(sim, zip3='850', cumulative=True, show_samples=10)
    name_zip850_c = _slug(fig_zip850_c.layout.title.text)
    fig_zip850_c.write_html(OUT_DIR / f"{name_zip850_c}.html")
    _safe_write_image(fig_zip850_c, OUT_DIR / f"{name_zip850_c}.png")

    # -----------------------------------------------------------------
    # Top-N ZIP3 bars for 2025 vs 2026 (based on PMF × total units)
    # -----------------------------------------------------------------
    def plot_top_zip(scenario="expected_median", sort_by="2026", top_n=20,
                     title="Top ZIP3 — comparison 2025 vs 2026"):
        col = f"units_daily__{scenario}"
        if col not in df_global.columns:
            raise KeyError(f"Column {col} not found in df_global.")

        total_2025 = units_dodba_current
        # You can choose another 2026 scenario for the bar chart; we use conservative here:
        total_2026 = units_2026_conservative

        z = df_zip3pmf.copy()
        z['zip3'] = z['zip3'].astype(str).str.zfill(3)
        z = z[['zip3', 'pmf']].dropna()

        z['dem_2025'] = z['pmf'] * total_2025
        z['dem_2026'] = z['pmf'] * total_2026
        z['dem_total'] = z['dem_2025'] + z['dem_2026']

        if sort_by == "2025":
            z = z.sort_values('dem_2025', ascending=False)
        elif sort_by == "total":
            z = z.sort_values('dem_total', ascending=False)
        else:
            z = z.sort_values('dem_2026', ascending=False)

        top = z.head(top_n).sort_values('dem_2026', ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top['zip3'], y=top['dem_2026'],
            name="2026",
            hovertemplate="ZIP3=%{x}<br>2026: %{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=top['zip3'], y=top['dem_2025'],
            name="2025",
            hovertemplate="ZIP3=%{x}<br>2025: %{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(
            title=title + f" — scenario: {scenario}",
            barmode="group",
            xaxis_title="ZIP3 (Top N)",
            yaxis_title="Units",
            yaxis_tickformat=",",
            bargap=0.25
        )
        return fig

    fig_top100 = plot_top_zip(scenario="expected_median", sort_by="2026", top_n=100,
                              title="Top 100 ZIP3 — comparison 2025 vs 2026")
    name_top100 = _slug(fig_top100.layout.title.text)
    fig_top100.write_html(OUT_DIR / f"{name_top100}.html")
    _safe_write_image(fig_top100, OUT_DIR / f"{name_top100}.png")

    # -----------------------------------------------------------------
    # Market/State breakdowns
    # -----------------------------------------------------------------
    dfm = df_zip3market.copy()
    dfm["zip3"] = dfm["zip3"].astype(str).str.zfill(3)

    # Count of ZIP3 per market
    by_market = (
        dfm.groupby("market", dropna=False)
           .size().reset_index(name="n_zip3")
           .sort_values("n_zip3", ascending=False)
    )
    fig_market = px.bar(by_market, x="market", y="n_zip3", text="n_zip3",
                        title="Count of ZIP3 by market")
    fig_market.update_layout(xaxis_title="Market", yaxis_title="Number of ZIP3")
    name_fig_market = _slug(fig_market.layout.title.text)
    fig_market.write_html(OUT_DIR / f"{name_fig_market}.html")
    _safe_write_image(fig_market, OUT_DIR / f"{name_fig_market}.png")

    # Count of ZIP3 by state
    by_state = (
        dfm.groupby("state", dropna=False)
           .size().reset_index(name="n_zip3")
           .sort_values("n_zip3", ascending=False)
    )
    fig_state = px.bar(by_state, x="state", y="n_zip3", text="n_zip3",
                       title="Count of ZIP3 by state")
    fig_state.update_layout(xaxis_title="State", yaxis_title="Number of ZIP3")
    name_fig_state = _slug(fig_state.layout.title.text)
    fig_state.write_html(OUT_DIR / f"{name_fig_state}.html")
    _safe_write_image(fig_state, OUT_DIR / f"{name_fig_state}.png")

    # Stacked market share within each state
    state_market = (
        dfm.groupby(["state", "market"], dropna=False)
           .size().reset_index(name="n_zip3")
    )
    fig_stack = px.bar(state_market, x="state", y="n_zip3", color="market",
                       barmode="stack", title="ZIP3 distribution by state × market")
    fig_stack.update_layout(xaxis_title="State", yaxis_title="Number of ZIP3")
    name_fig_stack = _slug(fig_stack.layout.title.text)
    fig_stack.write_html(OUT_DIR / f"{name_fig_stack}.html")
    _safe_write_image(fig_stack, OUT_DIR / f"{name_fig_stack}.png")

    # Demand-weighted by market and by state (using expected 2026 units)
    df_zip3work = df_zip3pmf.merge(df_zip3market, on='zip3')
    df_zip3work["units_2026"] = units_2026 * df_zip3work["pmf"]

    by_market_units = (
        df_zip3work.groupby("market", as_index=False)
                   .agg(units_2026=("units_2026", "sum"))
    )
    fig_market_units = px.bar(by_market_units, x="market", y="units_2026", text="units_2026",
                              title="Demand in 2026 by market")
    fig_market_units.update_layout(xaxis_title="Market", yaxis_title="Units 2026")
    fig_market_units.update_traces(texttemplate="%{text:,.0f}")
    fig_market_units.update_yaxes(tickformat=",")
    name_fig_market_units = _slug(fig_market_units.layout.title.text)
    fig_market_units.write_html(OUT_DIR / f"{name_fig_market_units}.html")
    _safe_write_image(fig_market_units, OUT_DIR / f"{name_fig_market_units}.png")

    by_state_units = (
        df_zip3work.groupby("state", as_index=False)
                   .agg(units_2026=("units_2026", "sum"))
    )
    fig_state_units = px.bar(by_state_units, x="state", y="units_2026", text="units_2026",
                             title="Demand in 2026 by state")
    fig_state_units.update_layout(xaxis_title="State", yaxis_title="Units 2026")
    fig_state_units.update_traces(texttemplate="%{text:,.0f}")
    fig_state_units.update_yaxes(tickformat=",")
    name_fig_state_units = _slug(fig_state_units.layout.title.text)
    fig_state_units.write_html(OUT_DIR / f"{name_fig_state_units}.html")
    _safe_write_image(fig_state_units, OUT_DIR / f"{name_fig_state_units}.png")

    state_market_units = (
        df_zip3work.groupby(["state", "market"], as_index=False)
                   .agg(units_2026=("units_2026", "sum"))
    )
    fig_stack_units = px.bar(state_market_units, x="state", y="units_2026", color="market",
                             barmode="stack", title="Demand by market and by state (2026)")
    fig_stack_units.update_layout(xaxis_title="State", yaxis_title="Units 2026")
    fig_stack_units.update_yaxes(tickformat=",")
    name_fig_stack_units = _slug(fig_stack_units.layout.title.text)
    fig_stack_units.write_html(OUT_DIR / f"{name_fig_stack_units}.html")
    _safe_write_image(fig_stack_units, OUT_DIR / f"{name_fig_stack_units}.png")

    # -----------------------------------------------------------------
    # Spatial annual distribution with error bars (top-K ZIP3 by PMF)
    # -----------------------------------------------------------------
    def _normalize_zip3pmf(df_zip3pmf: pd.DataFrame) -> pd.DataFrame:
        z = df_zip3pmf.copy()
        z['zip3'] = z['zip3'].astype(str).str.zfill(3)
        z['pmf'] = pd.to_numeric(z['pmf'], errors='coerce').fillna(0.0)
        s = z['pmf'].sum()
        if s > 0:
            z['pmf'] = z['pmf'] / s
        return z

    def _scenario_quantiles_year_end(sim_: dict, scen: str):
        pack = sim_[scen]
        q5 = float(pack['q5_cum'][-1])
        q50 = float(pack['q50_cum'][-1])
        q95 = float(pack['q95_cum'][-1])
        return q5, q50, q95

    def plot_spatial_year_dropdown(sim_, df_zip3pmf, top_k=25,
                                   title="Annual spatial distribution (ZIP3)"):
        z = _normalize_zip3pmf(df_zip3pmf)
        z_top = z.sort_values('pmf', ascending=False).head(top_k).copy()
        z_top = z_top.sort_values('pmf', ascending=True)  # horizontal bars read better

        scenarios_ = [s for s in ["expected", "optimistic", "conservative"] if s in sim_]
        if not scenarios_:
            raise ValueError("No scenarios found among ['expected','optimistic','conservative'].")

        fig = go.Figure()
        for i, scen in enumerate(scenarios_):
            q5, q50, q95 = _scenario_quantiles_year_end(sim_, scen)
            median = z_top['pmf'].to_numpy() * q50
            low = z_top['pmf'].to_numpy() * q5
            high = z_top['pmf'].to_numpy() * q95
            err_plus = high - median
            err_minus = median - low

            fig.add_trace(go.Bar(
                x=median, y=z_top['zip3'],
                orientation='h', name=scen,
                error_x=dict(type='data', array=err_plus, arrayminus=err_minus, visible=True),
                visible=(i == 0)
            ))

        buttons = []
        for i, scen in enumerate(scenarios_):
            vis = [False] * len(scenarios_)
            vis[i] = True
            buttons.append(dict(
                label=scen,
                method="update",
                args=[{"visible": vis},
                      {"title": f"{title} — scenario: {scen}",
                       "xaxis": {"title": "Units (median with 5–95% band)"},
                       "yaxis": {"title": "ZIP3"}}]
            ))
        fig.update_layout(
            title=f"{title} — scenario: {scenarios_[0]}",
            xaxis_title="Units (median with 5–95% band)",
            yaxis_title="ZIP3",
            updatemenus=[dict(type="dropdown", direction="down", x=1.02, xanchor="left", y=1, buttons=buttons)]
        )
        return fig

    fig_spatial = plot_spatial_year_dropdown(sim, df_zip3pmf, top_k=25)
    name_spatial = _slug(fig_spatial.layout.title.text)
    fig_spatial.write_html(OUT_DIR / f"{name_spatial}.html")
    _safe_write_image(fig_spatial, OUT_DIR / f"{name_spatial}.png")

    # -----------------------------------------------------------------
    # Final bookkeeping
    # -----------------------------------------------------------------
    # Save seasonality-weighted daily schedule
    df_global2.to_csv(OUT_DIR / "df_global2_seasonality_daily.csv", index=False)

    print("\n✅ Completed. All outputs are in:", OUT_DIR)
    print("   - HTML plots are viewable in any browser")
    print("   - PNG exports are written when 'kaleido' is installed (pip install kaleido)")
