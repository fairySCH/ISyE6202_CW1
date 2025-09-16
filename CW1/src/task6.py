# task 6 solution
# optimal OTD promise by market type to maximize profit, comparing them to the extremes

import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
# location of the input CSVs
BASE_DIR = Path(r"D:\Adarsh GATech Files\6335 Benoit SC1\CW1\ISyE6202_CW1\CW1")
DATA_DIR = BASE_DIR / "data"
TASK3_OUTPUT_DIR = BASE_DIR / "output" / "task3"
TASK5_OUTPUT_DIR = BASE_DIR / "output" / "task5"
OUTPUT_DIR = BASE_DIR / "output" / "task6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# prices
PRICE_PER_UNIT = 3000
COGS_PER_UNIT = 750

# load the data from said CSVs
def load_data():
    print("Loading CSVs...")
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
            exit()
    # correcting the columns in zip3_market.csv              
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

def solve_task_6(df_demand, df_assignment, network_name, df_conversion, df_shipping, df_task5_summary):
    print(f"Running optimization for {network_name} network...")
    # merge demand by zip table with zip assignment table. used gemini for dataframe operations
    df_full = pd.merge(df_demand, df_assignment, on='zip3')
    # merging with conversion rates and shipping costs
    df_expanded = pd.merge(df_full, df_conversion, left_on='market_type', right_on='Market')
    df_expanded = pd.merge(df_expanded, df_shipping, on=['distance_bucket', 'otd_promise'])
    # calculating converted demand and financials for each scenario
    df_expanded['converted_demand'] = df_expanded['demand_units'] * df_expanded['conversion_rate']
    df_expanded['total_revenue'] = df_expanded['converted_demand'] * PRICE_PER_UNIT
    df_expanded['total_shipping_cost'] = df_expanded['converted_demand'] * df_expanded['shipping_cost_per_unit']
    df_expanded['total_cogs'] = df_expanded['converted_demand'] * COGS_PER_UNIT
    df_expanded['gross_operating_profit'] = (df_expanded['total_revenue'] - df_expanded['total_shipping_cost'] - df_expanded['total_cogs'])
    # summing up financials by market type and otd promise
    financials_by_choice = df_expanded.groupby(['market_type', 'otd_promise']).agg(
        gross_operating_profit=('gross_operating_profit', 'sum'),
        total_revenue=('total_revenue', 'sum'),
        total_shipping_cost=('total_shipping_cost', 'sum'),
        total_cogs=('total_cogs', 'sum')
    ).reset_index()
    # greedy algorithm - for each market type, selects the otd promise that gives the highest profit
    optimal_choices = financials_by_choice.loc[financials_by_choice.groupby('market_type')['gross_operating_profit'].idxmax()]
    optimal_choices['OTD Promise (Days)'] = optimal_choices['otd_promise'].apply(lambda x: str(x) if x < 6 else '5+')
    optimal_choices['Network'] = network_name
    # sum the profit from greedy choices to find overall network profits
    optimized_profit = optimal_choices['gross_operating_profit'].sum()
    # profits for extremes (1 and 5+ otd)
    task5_network_summary = df_task5_summary[df_task5_summary['Network'] == network_name]
    profit_1_day_all = task5_network_summary[task5_network_summary['OTD Promise (Days)'] == '1']['Gross Operating Profit'].iloc[0]
    profit_5plus_day_all = task5_network_summary[task5_network_summary['OTD Promise (Days)'] == '5+']['Gross Operating Profit'].iloc[0]
    #  comparing to extremes
    comparison_data = {'Strategy': ['Optimized (Best for each Market)', '1-Day Promise for ALL', '5+ Day Promise for ALL'], 'Gross Operating Profit': [optimized_profit, profit_1_day_all, profit_5plus_day_all]}
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['Network'] = network_name
    
    return optimal_choices, comparison_df

def main():
    data_files = load_data()
    df_demand = pd.read_csv(TASK5_OUTPUT_DIR / "task5_demand_by_zip.csv")
    df_demand['zip3'] = df_demand['zip3'].astype(str).str.zfill(3)
    # given conversion rate data
    data_conversion = {'Market': ['Primary','Secondary','Tertiary'],1:[1.00,1.00,1.00],2:[0.90,1.00,1.00],3:[0.75,0.95,1.00],4:[0.60,0.75,0.95],5:[0.40,0.60,0.80],'5+':[0.30,0.40,0.60]}
    df_conversion = pd.DataFrame(data_conversion).melt(id_vars='Market', var_name='otd_promise', value_name='conversion_rate')
    df_conversion['otd_promise'] = pd.to_numeric(df_conversion['otd_promise'].replace('5+', 6))
    # given shipping cost data
    data_shipping = {'distance_bucket':['<50','51-150','151-300','301-600','601-1000','1001-1400','1401-1800','>1800'],1:[607,759,1025,1445,2078,2692,2841,2912],2:[353,441,585,794,924,1427,1795,1854],3:[230,287,392,533,655,895,1202,1330],4:[121,173,238,316,343,491,776,894],5:[139,151,191,242,287,340,362,388],'5+':[103,128,160,198,225,259,276,301]}
    df_shipping = pd.DataFrame(data_shipping).melt(id_vars='distance_bucket', var_name='otd_promise', value_name='shipping_cost_per_unit')
    # replace the 6 with 5+
    df_shipping['otd_promise'] = pd.to_numeric(df_shipping['otd_promise'].replace('5+', 6))

    # scenarios to evaluate based on network type
    scenarios = [
        (data_files["assign_1fc"], "1-FC"),
        (data_files["assign_4fc"], "4-FC"),
        (data_files["assign_15fc"], "15-FC"),
    ]
    all_optimal_choices, all_comparisons = [], []
    # iterate through each scenario
    for df_assignment, network_name in scenarios:
        df_assignment['zip3'] = df_assignment['zip3'].astype(str).str.zfill(3)
        optimal_df, comparison_df = solve_task_6(
            df_demand, df_assignment, network_name,
            df_conversion, df_shipping, data_files["task5_summary"]
        )
        all_optimal_choices.append(optimal_df)
        all_comparisons.append(comparison_df)
        
    final_optimal_choices = pd.concat(all_optimal_choices, ignore_index=True)
    final_comparisons = pd.concat(all_comparisons, ignore_index=True)

    print("Analysis complete. Saving primary outputs...")
    # used gemini for plots and CSV saving (including plotly syntax)
    # CSV 1: optimal otd promise by market type for each network
    optimal_choices_to_save = final_optimal_choices.set_index(['Network', 'market_type']).copy()
    optimal_choices_to_save['gross_operating_profit'] = optimal_choices_to_save['gross_operating_profit'].map('${:,.0f}'.format)
    optimal_policy_path = OUTPUT_DIR / "task6_optimal_policy.csv"
    optimal_choices_to_save[['OTD Promise (Days)', 'gross_operating_profit']].to_csv(optimal_policy_path)
    print(f"-> Optimal Policy table saved to: {optimal_policy_path}")

    # CSV 2: profit comparison across strategies for each network
    comparisons_to_save = final_comparisons.set_index(['Network', 'Strategy']).copy()
    comparisons_to_save['Gross Operating Profit'] = comparisons_to_save['Gross Operating Profit'].map('${:,.0f}'.format)
    comparison_path = OUTPUT_DIR / "task6_profit_comparison.csv"
    comparisons_to_save.to_csv(comparison_path)
    print(f"-> Profit Comparison table saved to: {comparison_path}")

    # CSV 3: detailed financials for the optimal policy
    detailed_optimal_path = OUTPUT_DIR / "task6_optimal_policy_details.csv"
    detailed_optimal_df = final_optimal_choices[['Network', 'market_type', 'OTD Promise (Days)', 'gross_operating_profit', 'total_revenue', 'total_shipping_cost', 'total_cogs']]
    detailed_optimal_df.to_csv(detailed_optimal_path, index=False)
    print(f"-> Detailed optimal policy financials saved to: {detailed_optimal_path}")

    # expected financials by market and overall for the optimal policy
    financials_summary = final_optimal_choices.groupby(['Network', 'market_type']).agg(
        Total_Revenue=('total_revenue', 'sum'),
        Total_Shipping_Cost=('total_shipping_cost', 'sum'),
        Total_COGS=('total_cogs', 'sum'),
        Gross_Operating_Profit=('gross_operating_profit', 'sum')
    ).reset_index()
    financials_summary['Net_Revenue'] = financials_summary['Total_Revenue'] - financials_summary['Total_Shipping_Cost']

    # calculating total revenue, net revenue and cogs for each network
    overall_summary = financials_summary.groupby('Network').agg(
        Total_Revenue=('Total_Revenue', 'sum'),
        Total_Shipping_Cost=('Total_Shipping_Cost', 'sum'),
        Total_COGS=('Total_COGS', 'sum'),
        Gross_Operating_Profit=('Gross_Operating_Profit', 'sum'),
        Net_Revenue=('Net_Revenue', 'sum')
    ).reset_index()
    overall_summary['market_type'] = 'Overall'

    # combining summaries
    financials_full = pd.concat([financials_summary, overall_summary], ignore_index=True)

    # CSV 4: comprehensive financial summary
    financials_csv_path = OUTPUT_DIR / "task6_optimal_policy_financials.csv"
    financials_full.to_csv(financials_csv_path, index=False)
    print(f"-> Optimal policy financials by market and overall saved to: {financials_csv_path}")


    # graph 1: max profit by network type
    max_profit_df = final_comparisons[final_comparisons['Strategy'] == 'Optimized (Best for each Market)'].sort_values('Gross Operating Profit', ascending=False)
    fig1 = px.bar(max_profit_df, x='Network', y='Gross Operating Profit', color='Network', title='Max Optimized Profit by Network Configuration', labels={'Gross Operating Profit': 'Max Gross Operating Profit ($)'})
    fig1_path = OUTPUT_DIR / "task6_viz_max_profit_by_network.png"; fig1.write_image(fig1_path, width=800, height=500); print(f"-> Max profit chart saved to: {fig1_path}")

    # graph 2: optimal otd promise by market and network type
    policy_df = final_optimal_choices.copy(); policy_df['OTD Promise (Numeric)'] = pd.to_numeric(policy_df['OTD Promise (Days)'].replace('5+', 6))
    fig2 = px.bar(policy_df, x='market_type', y='OTD Promise (Numeric)', color='Network', barmode='group', title='Optimal OTD Promise Policy by Market and Network', labels={'market_type': 'Market Type', 'OTD Promise (Numeric)': 'Optimal OTD Promise (Days)'}, category_orders={'market_type': ['Primary', 'Secondary', 'Tertiary']})
    fig2_path = OUTPUT_DIR / "task6_viz_optimal_policy.png"; fig2.write_image(fig2_path, width=800, height=500); print(f"-> Optimal policy chart saved to: {fig2_path}")

    # graph 3: profit contribution by market type for optimized strategy
    fig3 = px.bar(final_optimal_choices, x='Network', y='gross_operating_profit', color='market_type', title='Profit Contribution by Market Type (Optimized Strategy)', labels={'gross_operating_profit': 'Gross Operating Profit ($)', 'market_type': 'Market Type'}, category_orders={'market_type': ['Primary', 'Secondary', 'Tertiary']})
    fig3_path = OUTPUT_DIR / "task6_viz_profit_contribution.png"; fig3.write_image(fig3_path, width=800, height=500); print(f"-> Profit contribution chart saved to: {fig3_path}")
    
    task5_summary = data_files['task5_summary']; best_single_profits = task5_summary.groupby('Network')['Gross Operating Profit'].max(); comparison_for_plot = final_comparisons.set_index('Network'); comparison_for_plot['Best Single Profit'] = best_single_profits
    optimized_summary = comparison_for_plot[comparison_for_plot['Strategy'] == 'Optimized (Best for each Market)'].copy()
    optimized_summary['Profit Uplift (%)'] = ((optimized_summary['Gross Operating Profit'] / optimized_summary['Best Single Profit']) - 1) * 100
    
    # graph 4: comparing optimal profit to otd = 1 and 5+ strategies
    profit_comp = pd.read_csv(OUTPUT_DIR / "task6_profit_comparison.csv")

    # convert profit to float (in millions)
    profit_comp['Gross Operating Profit'] = profit_comp['Gross Operating Profit'].replace('[\$,]', '', regex=True).astype(float)
    profit_comp['Gross Operating Profit (M)'] = profit_comp['Gross Operating Profit'] / 1e6
    fig5 = px.bar(
        profit_comp,
        x='Network',
        y='Gross Operating Profit',
        color='Strategy',
        barmode='group',
        title='Gross Operating Profit by Strategy and FC Network',
        labels={'Gross Operating Profit': 'Gross Operating Profit ($)', 'Network': 'FC Network'},
        category_orders={'Strategy': ['Optimized (Best for each Market)', '1-Day Promise for ALL', '5+ Day Promise for ALL']}
    )
    fig5_path = OUTPUT_DIR / "task6_viz_profit_comparison_simple.png"
    fig5.write_image(fig5_path, width=1000, height=550)
    print(f"-> Simple profit comparison chart saved to: {fig5_path}")
    
    # graph 5: comprehenseive financial summary
    metrics = [
        ('Total_Revenue', 'Total Revenue ($)', 'task6_viz_optimal_total_revenue.png'),
        ('Total_Shipping_Cost', 'Total Shipping Cost ($)', 'task6_viz_optimal_total_shipping_cost.png'),
        ('Net_Revenue', 'Net Revenue ($)', 'task6_viz_optimal_net_revenue.png'),
        ('Gross_Operating_Profit', 'Gross Operating Profit ($)', 'task6_viz_optimal_gross_profit.png')
    ]
    for metric, label, filename in metrics:
        financials_full[metric + ' (M)'] = financials_full[metric] / 1e6
        fig = px.bar(
            financials_full,
            x='market_type',
            y=metric,
            color='Network',
            barmode='group',
            title=f'{label} by Market and Network (Optimal Policy)',
            labels={metric: label, 'market_type': 'Market Type'},
            category_orders={'market_type': ['Primary', 'Secondary', 'Tertiary', 'Overall']}
        )
        fig_path = OUTPUT_DIR / filename
        fig.write_image(fig_path, width=1000, height=550)
        print(f"-> {label} chart saved to: {fig_path}")

    print("\nScript finished successfully! 🎉")

if __name__ == "__main__":
    main()