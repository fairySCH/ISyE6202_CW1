# ==============================================================================
# ISYE 6202, Casework 1, Fall 2025
# Task 5 Solution Script - Final Definitive Version
#
# Description:
# This script performs the complete financial analysis for Task 5. It calculates
# Total Revenue, Total Shipping Cost, Net Revenue, Total COGS, and Gross
# Operating Profit for three network scenarios across all OTD promises,
# and generates all required outputs.
# ==============================================================================

# --- 1. SETUP AND CONFIGURATION ---

import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(r"D:\Adarsh GATech Files\6335 Benoit SC1\CW1\ISyE6202_CW1\CW1")
DATA_DIR = BASE_DIR / "data"
TASK3_OUTPUT_DIR = BASE_DIR / "output" / "task3"
OUTPUT_DIR = BASE_DIR / "output" / "task5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Financial Constants from the Casework ---
PRICE_PER_UNIT = 3000
COGS_PER_UNIT = 750

def load_data():
    """
    Loads all necessary input CSV files and corrects the market_type data.
    """
    print("Loading input data...")
    paths = {
        "seasonalities": DATA_DIR / "demand_seasonalities.csv",
        "zip3_pmf": DATA_DIR / "zip3_pmf.csv",
        "zip3_market": DATA_DIR / "zip3_market.csv",
        "assign_1fc": TASK3_OUTPUT_DIR / "assignment_1FC.csv",
        "assign_4fc": TASK3_OUTPUT_DIR / "assignment_4FC.csv",
        "assign_15fc": TASK3_OUTPUT_DIR / "assignment_15FC.csv",
    }
    data = {}
    for name, path in paths.items():
        try:
            data[name] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"--- ERROR: Input file not found at {path} ---")
            exit()
            
    correct_markets = data['zip3_market']
    correct_markets.columns = [c.strip().lower() for c in correct_markets.columns]
    correct_markets['zip3'] = correct_markets['zip3'].astype(str).str.zfill(3)
    correct_markets['market'] = correct_markets['market'].str.title()
    correct_markets.rename(columns={'market': 'market_type'}, inplace=True)
    
    for key in ["assign_1fc", "assign_4fc", "assign_15fc"]:
        data[key]['zip3'] = data[key]['zip3'].astype(str).str.zfill(3)
        if 'market_type' in data[key].columns:
            data[key] = data[key].drop(columns=['market_type'])
        data[key] = pd.merge(data[key], correct_markets[['zip3', 'market_type']], on='zip3', how='left')
    print("Data loaded and corrected successfully.")
    return data

def prepare_demand_data(df_seasonalities, df_zip3_pmf):
    """
    Calculates the master demand DataFrame (demand per ZIP, per day).
    """
    print("Preparing daily demand data by ZIP code...")
    total_units_overall = 2_000_000; overall_market_grow = 0.075; current_market_share = 0.036; median_market_share_grow_yearly = 0.20; N_DAYS = 364
    total_dobda_units_yearly = (total_units_overall * (1 + overall_market_grow)) * (current_market_share * (1 + median_market_share_grow_yearly))
    df_seasonalities.columns = [c.strip().lower().replace(' ', '_') for c in df_seasonalities.columns]
    df_season_week = df_seasonalities[["week_of_year", "proportion"]].copy(); df_day_week = df_seasonalities[["day_of_week", "proportion.1"]].iloc[:7].copy()
    df_season_week.loc[:, 'proportion'] = df_season_week['proportion'].str.replace('%', '').astype(float) / 100
    df_day_week.loc[:, 'proportion.1'] = df_day_week['proportion.1'].str.replace('%', '').astype(float) / 100
    dates = pd.to_datetime(pd.to_timedelta(np.arange(N_DAYS), unit='D') + pd.Timestamp('2026-01-01'))
    df_calendar = pd.DataFrame({'date': dates}); df_calendar['week_of_year'] = df_calendar['date'].dt.isocalendar().week; df_calendar['day_of_week'] = df_calendar['date'].dt.isocalendar().day
    df_calendar = pd.merge(df_calendar, df_season_week, on='week_of_year', how='left'); df_calendar = pd.merge(df_calendar, df_day_week.rename(columns={'proportion.1': 'day_proportion'}), on='day_of_week', how='left')
    df_calendar['pmf_daily'] = (df_calendar['proportion'] * df_calendar['day_proportion']).fillna(0); df_calendar['pmf_daily'] /= df_calendar['pmf_daily'].sum()
    df_calendar['total_us_demand'] = total_dobda_units_yearly * df_calendar['pmf_daily']
    df_zip3_pmf.columns = [c.strip().lower() for c in df_zip3_pmf.columns]; df_zip3_pmf['zip3'] = df_zip3_pmf['zip3'].astype(str).str.zfill(3)
    df_zip3_pmf['pmf'] /= df_zip3_pmf['pmf'].sum()
    df_demand = pd.merge(df_calendar[['date', 'total_us_demand']], df_zip3_pmf, how='cross')
    df_demand['demand_units'] = df_demand['total_us_demand'] * df_demand['pmf']
    print("Demand data preparation complete.")
    return df_demand[['date', 'zip3', 'demand_units']]

def calculate_financials_for_scenario(df_demand, df_assignment, network_name, df_conversion, df_shipping):
    """
    Calculates all required financial metrics for a given network across all OTD options.
    """
    print(f"Calculating financials for {network_name} network...")
    df_full = pd.merge(df_demand, df_assignment, on='zip3')
    results = []
    
    for otd in range(1, 7):
        df_scenario = df_full.copy()
        
        # Merge conversion rates and shipping costs
        df_scenario = pd.merge(df_scenario, df_conversion[df_conversion['otd_promise'] == otd], left_on='market_type', right_on='Market')
        df_scenario = pd.merge(df_scenario, df_shipping[df_shipping['otd_promise'] == otd], on='distance_bucket')
        
        # Calculate all financial metrics as requested in the prompt
        df_scenario['converted_demand'] = df_scenario['demand_units'] * df_scenario['conversion_rate']
        df_scenario['total_revenue'] = df_scenario['converted_demand'] * PRICE_PER_UNIT
        df_scenario['total_shipping_cost'] = df_scenario['converted_demand'] * df_scenario['shipping_cost_per_unit']
        df_scenario['net_revenue'] = df_scenario['total_revenue'] - df_scenario['total_shipping_cost']
        df_scenario['total_cogs'] = df_scenario['converted_demand'] * COGS_PER_UNIT
        df_scenario['gross_operating_profit'] = df_scenario['net_revenue'] - df_scenario['total_cogs']
        
        # Aggregate the results for the year and store them
        results.append({
            'Network': network_name,
            'OTD Promise (Days)': str(otd) if otd < 6 else '5+',
            'Total Revenue': df_scenario['total_revenue'].sum(),
            'Total Shipping Cost': df_scenario['total_shipping_cost'].sum(),
            'Net Revenue': df_scenario['net_revenue'].sum(),
            'Total COGS': df_scenario['total_cogs'].sum(),
            'Gross Operating Profit': df_scenario['gross_operating_profit'].sum()
        })
        
    return pd.DataFrame(results)

def main():
    """
    Main function to execute the complete Task 5 analysis.
    """
    data_files = load_data()
    df_demand = prepare_demand_data(data_files["seasonalities"], data_files["zip3_pmf"])

    print("Defining financial models...")
    data_conversion = {'Market': ['Primary','Secondary','Tertiary'],1:[1.00,1.00,1.00],2:[0.90,1.00,1.00],3:[0.75,0.95,1.00],4:[0.60,0.75,0.95],5:[0.40,0.60,0.80],'5+':[0.30,0.40,0.60]}
    df_conversion = pd.DataFrame(data_conversion).melt(id_vars='Market', var_name='otd_promise', value_name='conversion_rate')
    df_conversion['otd_promise'] = pd.to_numeric(df_conversion['otd_promise'].replace('5+', 6))
    data_shipping = {'distance_bucket':['<50','51-150','151-300','301-600','601-1000','1001-1400','1401-1800','>1800'],1:[607,759,1025,1445,2078,2692,2841,2912],2:[353,441,585,794,924,1427,1795,1854],3:[230,287,392,533,655,895,1202,1330],4:[121,173,238,316,343,491,776,894],5:[139,151,191,242,287,340,362,388],'5+':[103,128,160,198,225,259,276,301]}
    df_shipping = pd.DataFrame(data_shipping).melt(id_vars='distance_bucket', var_name='otd_promise', value_name='shipping_cost_per_unit')
    df_shipping['otd_promise'] = pd.to_numeric(df_shipping['otd_promise'].replace('5+', 6))

    scenarios = [
        (data_files["assign_1fc"], "1-FC"),
        (data_files["assign_4fc"], "4-FC"),
        (data_files["assign_15fc"], "15-FC"),
    ]
    
    all_results = [calculate_financials_for_scenario(df_demand, df, name, df_conversion, df_shipping) for df, name in scenarios]
    final_summary_df = pd.concat(all_results, ignore_index=True)

    print("\nAnalysis complete. Saving primary outputs...")
    summary_csv_path = OUTPUT_DIR / "task5_financial_summary.csv"
    final_summary_df.to_csv(summary_csv_path, index=False)
    print(f"-> Main summary table saved to: {summary_csv_path}")
    
    # --- Primary Visualization: Gross Operating Profit ---
    fig = px.bar(final_summary_df,
                 x='OTD Promise (Days)', y='Gross Operating Profit', color='Network',
                 barmode='group', title='Task 5: Gross Operating Profit by OTD Promise and Network',
                 labels={'Gross Operating Profit': 'Gross Operating Profit ($)'},
                 category_orders={"OTD Promise (Days)": ["1", "2", "3", "4", "5", "5+"]})
    
    chart_png_path = OUTPUT_DIR / "task5_profit_by_otd_chart.png"
    try:
        fig.write_image(chart_png_path, width=1000, height=550)
        print(f"-> Main profit chart saved to: {chart_png_path}")
    except Exception:
        print("\n--- Could not save image file. Please install kaleido by running: pip install kaleido ---")

    # --- Additional Visualizations and Outputs ---
    print("\nGenerating additional outputs...")

    # Viz 1: Revenue vs. Costs Trade-off (Now including Net Revenue)
    fig_tradeoff = px.line(
        final_summary_df, x='OTD Promise (Days)', y=['Total Revenue', 'Total Shipping Cost', 'Net Revenue'],
        facet_row='Network', title='Task 5: The Trade-Off Between Revenue, Costs, and Net Revenue',
        labels={'value': 'Amount ($)', 'variable': 'Metric'}, markers=True
    )
    tradeoff_chart_path = OUTPUT_DIR / "task5_tradeoff_chart.png"
    fig_tradeoff.write_image(tradeoff_chart_path, width=1000, height=800)
    print(f"-> Trade-off chart saved to: {tradeoff_chart_path}")
    
    # Viz 2: Detailed Financial Breakdown
    df_melted = final_summary_df.melt(
        id_vars=['Network', 'OTD Promise (Days)'],
        value_vars=['Net Revenue', 'Total COGS'],
        var_name='Metric', value_name='Amount'
    )
    fig_breakdown = px.bar(
        df_melted, x='OTD Promise (Days)', y='Amount', color='Metric',
        facet_row='Network', title='Task 5: Financial Breakdown (Net Revenue vs. COGS)',
        labels={'Amount': 'Amount ($)'}, category_orders={"OTD Promise (Days)": ["1", "2", "3", "4", "5", "5+"]}
    )
    breakdown_chart_path = OUTPUT_DIR / "task5_profit_breakdown_chart.png"
    fig_breakdown.write_image(breakdown_chart_path, width=1200, height=800)
    print(f"-> Financial breakdown chart saved to: {breakdown_chart_path}")

    # Additional CSV: Profit by Market Type for the 4-FC Network
    print("Generating detailed profit breakdown by market type (for 4-FC network)...")
    df_expanded = pd.merge(df_demand, data_files["assign_4fc"], on='zip3')
    df_expanded = pd.merge(df_expanded, df_conversion, left_on='market_type', right_on='Market')
    df_expanded = pd.merge(df_expanded, df_shipping, on=['distance_bucket', 'otd_promise'])
    df_expanded['gross_operating_profit'] = ((df_expanded['demand_units'] * df_expanded['conversion_rate']) * PRICE_PER_UNIT) - \
                                            ((df_expanded['demand_units'] * df_expanded['conversion_rate']) * df_expanded['shipping_cost_per_unit']) - \
                                            ((df_expanded['demand_units'] * df_expanded['conversion_rate']) * COGS_PER_UNIT)
    market_profit_by_otd = df_expanded.groupby(['market_type', 'otd_promise'])['gross_operating_profit'].sum().reset_index()
    market_profit_by_otd['OTD Promise (Days)'] = market_profit_by_otd['otd_promise'].apply(lambda x: str(x) if x < 6 else '5+')
    market_breakdown_csv_path = OUTPUT_DIR / "task5_profit_by_market_type_(4-FC).csv"
    market_profit_by_otd[['market_type', 'OTD Promise (Days)', 'gross_operating_profit']].to_csv(market_breakdown_csv_path, index=False)
    print(f"-> Profit breakdown by market type saved to: {market_breakdown_csv_path}")

    print("\nScript finished successfully! 🎉")

if __name__ == "__main__":
    main()