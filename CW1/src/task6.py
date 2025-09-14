# ==============================================================================
# ISYE 6202, Casework 1, Fall 2025
# Task 6 Solution Script - Final Definitive Version
#
# Description:
# This script proposes an optimized Order-to-Delivery (OTD) time promise
# for each market type to maximize profit. It computes the detailed financials
# for this optimal strategy and contrasts it with extreme scenarios.
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
TASK5_OUTPUT_DIR = BASE_DIR / "output" / "task5"
OUTPUT_DIR = BASE_DIR / "output" / "task6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Financial Constants ---
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
        "task5_summary": TASK5_OUTPUT_DIR / "task5_financial_summary.csv",
    }
    data = {}
    for name, path in paths.items():
        try:
            data[name] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"--- ERROR: Input file not found at {path} ---")
            print("Please make sure Task 5 has been run and its output CSV exists.")
            exit()
            
    correct_markets = data['zip3_market'].copy()
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
    print("Preparing daily demand data...")
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
    return df_demand[['date', 'zip3', 'demand_units']]

def solve_task_6(df_demand, df_assignment, network_name, df_conversion, df_shipping, df_task5_summary):
    """
    Finds the optimal OTD promise for each market type to maximize total profit
    and compares this strategy to the extremes.
    """
    print(f"Running optimization for {network_name} network...")
    
    df_full = pd.merge(df_demand, df_assignment, on='zip3')
    df_expanded = pd.merge(df_full, df_conversion, left_on='market_type', right_on='Market')
    df_expanded = pd.merge(df_expanded, df_shipping, on=['distance_bucket', 'otd_promise'])
    
    df_expanded['converted_demand'] = df_expanded['demand_units'] * df_expanded['conversion_rate']
    df_expanded['total_revenue'] = df_expanded['converted_demand'] * PRICE_PER_UNIT
    df_expanded['total_shipping_cost'] = df_expanded['converted_demand'] * df_expanded['shipping_cost_per_unit']
    df_expanded['total_cogs'] = df_expanded['converted_demand'] * COGS_PER_UNIT
    df_expanded['gross_operating_profit'] = (df_expanded['total_revenue'] - df_expanded['total_shipping_cost'] - df_expanded['total_cogs'])
    
    financials_by_choice = df_expanded.groupby(['market_type', 'otd_promise']).agg(
        gross_operating_profit=('gross_operating_profit', 'sum'),
        total_revenue=('total_revenue', 'sum'),
        total_shipping_cost=('total_shipping_cost', 'sum'),
        total_cogs=('total_cogs', 'sum')
    ).reset_index()
    
    optimal_choices = financials_by_choice.loc[financials_by_choice.groupby('market_type')['gross_operating_profit'].idxmax()]
    optimal_choices['OTD Promise (Days)'] = optimal_choices['otd_promise'].apply(lambda x: str(x) if x < 6 else '5+')
    optimal_choices['Network'] = network_name
    
    optimized_profit = optimal_choices['gross_operating_profit'].sum()
    task5_network_summary = df_task5_summary[df_task5_summary['Network'] == network_name]
    profit_1_day_all = task5_network_summary[task5_network_summary['OTD Promise (Days)'] == '1']['Gross Operating Profit'].iloc[0]
    profit_5plus_day_all = task5_network_summary[task5_network_summary['OTD Promise (Days)'] == '5+']['Gross Operating Profit'].iloc[0]

    comparison_data = {'Strategy': ['Optimized (Best for each Market)', '1-Day Promise for ALL', '5+ Day Promise for ALL'], 'Gross Operating Profit': [optimized_profit, profit_1_day_all, profit_5plus_day_all]}
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Network'] = network_name
    
    return optimal_choices, comparison_df

def main():
    """
    Main function to execute the Task 6 analysis.
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

    scenarios = [(data_files["assign_1fc"], "1-FC"), (data_files["assign_4fc"], "4-FC"), (data_files["assign_15fc"], "15-FC")]
    all_optimal_choices, all_comparisons = [], []

    for df_assignment, network_name in scenarios:
        optimal_df, comparison_df = solve_task_6(df_demand, df_assignment, network_name, df_conversion, df_shipping, data_files["task5_summary"])
        all_optimal_choices.append(optimal_df)
        all_comparisons.append(comparison_df)
        
    # Create final dataframes with raw numbers for plotting
    final_optimal_choices = pd.concat(all_optimal_choices)
    final_comparisons = pd.concat(all_comparisons)
    
    print("Analysis complete. Saving primary outputs...")
    # Create FORMATTED copies for saving to CSV
    optimal_choices_to_save = final_optimal_choices.set_index(['Network', 'market_type']).copy()
    optimal_choices_to_save['gross_operating_profit'] = optimal_choices_to_save['gross_operating_profit'].map('${:,.0f}'.format)
    optimal_policy_path = OUTPUT_DIR / "task6_optimal_policy.csv"
    optimal_choices_to_save[['OTD Promise (Days)', 'gross_operating_profit']].to_csv(optimal_policy_path)
    print(f"-> Optimal Policy table saved to: {optimal_policy_path}")

    comparisons_to_save = final_comparisons.set_index(['Network', 'Strategy']).copy()
    comparisons_to_save['Gross Operating Profit'] = comparisons_to_save['Gross Operating Profit'].map('${:,.0f}'.format)
    comparison_path = OUTPUT_DIR / "task6_profit_comparison.csv"
    comparisons_to_save.to_csv(comparison_path)
    print(f"-> Profit Comparison table saved to: {comparison_path}")

    # --- Additional CSV and Visualizations for Task 6 ---
    print("\nGenerating additional outputs for Task 6...")

    detailed_optimal_path = OUTPUT_DIR / "task6_optimal_policy_details.csv"
    detailed_optimal_df = final_optimal_choices[['Network', 'market_type', 'OTD Promise (Days)', 'gross_operating_profit', 'total_revenue', 'total_shipping_cost', 'total_cogs']]
    detailed_optimal_df.to_csv(detailed_optimal_path, index=False)
    print(f"-> Detailed optimal policy financials saved to: {detailed_optimal_path}")

    max_profit_df = final_comparisons[final_comparisons['Strategy'] == 'Optimized (Best for each Market)'].sort_values('Gross Operating Profit', ascending=False)
    fig1 = px.bar(max_profit_df, x='Network', y='Gross Operating Profit', color='Network', title='Max Optimized Profit by Network Configuration', labels={'Gross Operating Profit': 'Max Gross Operating Profit ($)'})
    fig1_path = OUTPUT_DIR / "task6_viz_max_profit_by_network.png"; fig1.write_image(fig1_path, width=800, height=500); print(f"-> Max profit chart saved to: {fig1_path}")

    policy_df = final_optimal_choices.copy(); policy_df['OTD Promise (Numeric)'] = pd.to_numeric(policy_df['OTD Promise (Days)'].replace('5+', 6))
    fig2 = px.bar(policy_df, x='market_type', y='OTD Promise (Numeric)', color='Network', barmode='group', title='Optimal OTD Promise Policy by Market and Network', labels={'market_type': 'Market Type', 'OTD Promise (Numeric)': 'Optimal OTD Promise (Days)'}, category_orders={'market_type': ['Primary', 'Secondary', 'Tertiary']})
    fig2_path = OUTPUT_DIR / "task6_viz_optimal_policy.png"; fig2.write_image(fig2_path, width=800, height=500); print(f"-> Optimal policy chart saved to: {fig2_path}")

    fig3 = px.bar(final_optimal_choices, x='Network', y='gross_operating_profit', color='market_type', title='Profit Contribution by Market Type (Optimized Strategy)', labels={'gross_operating_profit': 'Gross Operating Profit ($)', 'market_type': 'Market Type'}, category_orders={'market_type': ['Primary', 'Secondary', 'Tertiary']})
    fig3_path = OUTPUT_DIR / "task6_viz_profit_contribution.png"; fig3.write_image(fig3_path, width=800, height=500); print(f"-> Profit contribution chart saved to: {fig3_path}")
    
    task5_summary = data_files['task5_summary']; best_single_profits = task5_summary.groupby('Network')['Gross Operating Profit'].max(); comparison_for_plot = final_comparisons.set_index('Network'); comparison_for_plot['Best Single Profit'] = best_single_profits
    optimized_summary = comparison_for_plot[comparison_for_plot['Strategy'] == 'Optimized (Best for each Market)'].copy()
    optimized_summary['Profit Uplift (%)'] = ((optimized_summary['Gross Operating Profit'] / optimized_summary['Best Single Profit']) - 1) * 100
    fig4 = px.bar(optimized_summary.reset_index(), x='Network', y='Profit Uplift (%)', color='Network', title='Value of Optimization: Profit Uplift vs. Best Single Policy', labels={'Profit Uplift (%)': 'Profit Increase from Optimization (%)'})
    fig4_path = OUTPUT_DIR / "task6_viz_optimization_value.png"; fig4.write_image(fig4_path, width=800, height=500); print(f"-> Optimization value chart saved to: {fig4_path}")
    
    print("\nScript finished successfully! 🎉")

if __name__ == "__main__":
    main()