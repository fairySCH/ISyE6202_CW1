# task 5 solution
# total revenue, total shipping cost, net revenue, total cogs and gross operating profit calculation for 1-FC, 4-FC and 15-FC networks and primary secondary and tertiary markets

import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
# location of the input CSVs
BASE_DIR = Path(r"D:\Adarsh GATech Files\6335 Benoit SC1\CW1\ISyE6202_CW1\CW1")
DATA_DIR = BASE_DIR / "data"
TASK3_OUTPUT_DIR = BASE_DIR / "output" / "task3"
OUTPUT_DIR = BASE_DIR / "output" / "task5"
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
    }
    data = {}
    for name, path in paths.items():
        try:
            data[name] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"ERROR: Input file not found at {path}")
            exit()
    # correcting the columns in zip3_market.csv    
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

# preparing the demand data by ZIP Code (dataframe with demand per ZIP, per day). used Gemini's syntax approach for dataframe operations
def prepare_demand_data(df_seasonalities, df_zip3_pmf):
    print("Preparing table of demand per ZIP per day")

    # market size and dobda's share
    total_units_overall = 2_000_000
    overall_market_grow = 0.075
    current_market_share = 0.036
    median_market_share_grow_yearly = 0.20
    N_DAYS = 364

    # calculating dobda's expected market share
    market_size_2026 = total_units_overall * (1 + overall_market_grow)
    dobda_share_2026 = current_market_share * (1 + median_market_share_grow_yearly)
    total_dobda_units_yearly = market_size_2026 * dobda_share_2026

    # standardizing column names 
    df_seasonalities.columns = [c.strip().lower().replace(' ', '_') for c in df_seasonalities.columns]
    df_zip3_pmf.columns = [c.strip().lower() for c in df_zip3_pmf.columns]
    
    # converting daily and weekly seasonility from str to float
    df_season_week = df_seasonalities[["week_of_year", "proportion"]].copy()
    df_day_week = df_seasonalities[["day_of_week", "proportion.1"]].iloc[:7].copy()
    df_season_week.loc[:, 'proportion'] = df_season_week['proportion'].str.replace('%', '').astype(float) / 100
    df_day_week.loc[:, 'proportion.1'] = df_day_week['proportion.1'].str.replace('%', '').astype(float) / 100
    
    # Standardize the geographic demand data and ensure it sums to 100%
    df_zip3_pmf['zip3'] = df_zip3_pmf['zip3'].astype(str).str.zfill(3)
    df_zip3_pmf['pmf'] /= df_zip3_pmf['pmf'].sum()

    # for 364 days in 2026 this maps the right seasonal factor to each day
    dates = pd.to_datetime(pd.to_timedelta(np.arange(N_DAYS), unit='D') + pd.Timestamp('2026-01-01'))
    df_calendar = pd.DataFrame({'date': dates})
    df_calendar['week_of_year'] = df_calendar['date'].dt.isocalendar().week
    df_calendar['day_of_week'] = df_calendar['date'].dt.isocalendar().day
    df_calendar = pd.merge(df_calendar, df_season_week, on='week_of_year', how='left')
    df_calendar = pd.merge(df_calendar, df_day_week.rename(columns={'proportion.1': 'day_proportion'}), on='day_of_week', how='left')

    # calculating a unique demand weight for each day based on its corresponding seasonal factors
    df_calendar['pmf_daily'] = (df_calendar['proportion'] * df_calendar['day_proportion']).fillna(0)
    df_calendar['pmf_daily'] /= df_calendar['pmf_daily'].sum() # Normalize to ensure all days sum to 100%
    
    # distribute demand over 364 days
    df_calendar['total_us_demand'] = total_dobda_units_yearly * df_calendar['pmf_daily']

    # merging the daily demand with zip3_pmf to get demand per ZIP code per day
    df_demand = pd.merge(df_calendar[['date', 'total_us_demand']], df_zip3_pmf, how='cross')
    df_demand['demand_units'] = df_demand['total_us_demand'] * df_demand['pmf']
    print("Demand data preparation complete.")
    return df_demand[['date', 'zip3', 'demand_units']]

# calculating the financials based on market type and distance bucket. used Gemini for dataframe operations
def calculate_financials_for_scenario(df_demand, df_assignment, network_name, df_conversion, df_shipping):
    print(f"Calculating financials for {network_name} network...")
    
    df_full = pd.merge(df_demand, df_assignment, on='zip3')
    results = []
    # iterate over each OTD promise scenario (otd = 6 means 5+ days)
    for otd in range(1, 7):
        # current scenario dataframe with the generated df_full table
        df_scenario = df_full.copy()
        # merge with conversion rates and shipping costs based on the iterative otd value
        df_scenario = pd.merge(df_scenario, df_conversion[df_conversion['otd_promise'] == otd], left_on='market_type', right_on='Market')
        df_scenario = pd.merge(df_scenario, df_shipping[df_shipping['otd_promise'] == otd], on='distance_bucket')
        # calculation of all financials
        df_scenario['converted_demand'] = df_scenario['demand_units'] * df_scenario['conversion_rate']
        df_scenario['total_revenue'] = df_scenario['converted_demand'] * PRICE_PER_UNIT
        df_scenario['total_shipping_cost'] = df_scenario['converted_demand'] * df_scenario['shipping_cost_per_unit']
        df_scenario['net_revenue'] = df_scenario['total_revenue'] - df_scenario['total_shipping_cost']
        df_scenario['total_cogs'] = df_scenario['converted_demand'] * COGS_PER_UNIT
        df_scenario['gross_operating_profit'] = df_scenario['net_revenue'] - df_scenario['total_cogs']
        
        # aggregate and store results
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

# main function calls the functions and generates output CSVs and plots
def main():
    data_files = load_data()
    df_demand = prepare_demand_data(data_files["seasonalities"], data_files["zip3_pmf"])
    
    # given conversion rate data
    data_conversion = {'Market': ['Primary','Secondary','Tertiary'],1:[1.00,1.00,1.00],2:[0.90,1.00,1.00],3:[0.75,0.95,1.00],4:[0.60,0.75,0.95],5:[0.40,0.60,0.80],'5+':[0.30,0.40,0.60]}
    df_conversion = pd.DataFrame(data_conversion).melt(id_vars='Market', var_name='otd_promise', value_name='conversion_rate')
    df_conversion['otd_promise'] = pd.to_numeric(df_conversion['otd_promise'].replace('5+', 6))
    # given shipping cost data
    data_shipping = {'distance_bucket':['<50','51-150','151-300','301-600','601-1000','1001-1400','1401-1800','>1800'],1:[607,759,1025,1445,2078,2692,2841,2912],2:[353,441,585,794,924,1427,1795,1854],3:[230,287,392,533,655,895,1202,1330],4:[121,173,238,316,343,491,776,894],5:[139,151,191,242,287,340,362,388],'5+':[103,128,160,198,225,259,276,301]}
    df_shipping = pd.DataFrame(data_shipping).melt(id_vars='distance_bucket', var_name='otd_promise', value_name='shipping_cost_per_unit')
    # replace the 6 with 5+
    df_shipping['otd_promise'] = pd.to_numeric(df_shipping['otd_promise'].replace('5+', 6))

    scenarios = [
        (data_files["assign_1fc"], "1-FC"),
        (data_files["assign_4fc"], "4-FC"),
        (data_files["assign_15fc"], "15-FC"),
    ]
    
    # compute results for all scenarios and concatenate them
    all_results = [calculate_financials_for_scenario(df_demand, df, name, df_conversion, df_shipping) for df, name in scenarios]
    final_summary_df = pd.concat(all_results, ignore_index=True)

    # saving the final summary in CSV format along w graphs, used gemini for saving the images and CSVs to specific directory location
    # CSV 1: main summary table
    print("\nAnalysis complete. Saving CSVs and graphs...")
    summary_csv_path = OUTPUT_DIR / "task5_financial_summary.csv"
    final_summary_df.to_csv(summary_csv_path, index=False)
    print(f"-> Main summary table saved to: {summary_csv_path}")
    
    # CSV 2: save demand per ZIP per day for task 6 use
    print("Saving demand per ZIP per day table...")
    demand_csv_path = OUTPUT_DIR / "task5_demand_by_zip.csv"
    df_demand.to_csv(demand_csv_path, index=False)
    print(f"-> Demand table saved to: {demand_csv_path}")

    # CSV 3: detailed profit breakdown by market type (for 1-FC network)
    print("Generating detailed profit breakdown by market type (for 1-FC network)...")
    df_expanded_1fc = pd.merge(df_demand, data_files["assign_1fc"], on='zip3')
    df_expanded_1fc = pd.merge(df_expanded_1fc, df_conversion, left_on='market_type', right_on='Market')
    df_expanded_1fc = pd.merge(df_expanded_1fc, df_shipping, on=['distance_bucket', 'otd_promise'])
    df_expanded_1fc['gross_operating_profit'] = ((df_expanded_1fc['demand_units'] * df_expanded_1fc['conversion_rate']) * PRICE_PER_UNIT) - \
                                                ((df_expanded_1fc['demand_units'] * df_expanded_1fc['conversion_rate']) * df_expanded_1fc['shipping_cost_per_unit']) - \
                                                ((df_expanded_1fc['demand_units'] * df_expanded_1fc['conversion_rate']) * COGS_PER_UNIT)
    market_profit_by_otd_1fc = df_expanded_1fc.groupby(['market_type', 'otd_promise'])['gross_operating_profit'].sum().reset_index()
    market_profit_by_otd_1fc['OTD Promise (Days)'] = market_profit_by_otd_1fc['otd_promise'].apply(lambda x: str(x) if x < 6 else '5+')
    market_breakdown_csv_path_1fc = OUTPUT_DIR / "task5_profit_by_market_type_(1-FC).csv"
    market_profit_by_otd_1fc[['market_type', 'OTD Promise (Days)', 'gross_operating_profit']].to_csv(market_breakdown_csv_path_1fc, index=False)
    print(f"-> Profit breakdown by market type saved to: {market_breakdown_csv_path_1fc}")

  
    # CSV 4: detailed profit breakdown by market type (for 4-FC network)
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

    # CSV 5: detailed profit breakdown by market type (for 15-FC network)
    print("Generating detailed profit breakdown by market type (for 15-FC network)...")
    df_expanded_15fc = pd.merge(df_demand, data_files["assign_15fc"], on='zip3')
    df_expanded_15fc = pd.merge(df_expanded_15fc, df_conversion, left_on='market_type', right_on='Market')
    df_expanded_15fc = pd.merge(df_expanded_15fc, df_shipping, on=['distance_bucket', 'otd_promise'])
    df_expanded_15fc['gross_operating_profit'] = ((df_expanded_15fc['demand_units'] * df_expanded_15fc['conversion_rate']) * PRICE_PER_UNIT) - \
                                                 ((df_expanded_15fc['demand_units'] * df_expanded_15fc['conversion_rate']) * df_expanded_15fc['shipping_cost_per_unit']) - \
                                                 ((df_expanded_15fc['demand_units'] * df_expanded_15fc['conversion_rate']) * COGS_PER_UNIT)
    market_profit_by_otd_15fc = df_expanded_15fc.groupby(['market_type', 'otd_promise'])['gross_operating_profit'].sum().reset_index()
    market_profit_by_otd_15fc['OTD Promise (Days)'] = market_profit_by_otd_15fc['otd_promise'].apply(lambda x: str(x) if x < 6 else '5+')
    market_breakdown_csv_path_15fc = OUTPUT_DIR / "task5_profit_by_market_type_(15-FC).csv"
    market_profit_by_otd_15fc[['market_type', 'OTD Promise (Days)', 'gross_operating_profit']].to_csv(market_breakdown_csv_path_15fc, index=False)
    print(f"-> Profit breakdown by market type saved to: {market_breakdown_csv_path_15fc}") 

    # graph 1: gross operating profit by otd value and network type
    fig = px.bar(
        final_summary_df,
        x='OTD Promise (Days)',
        y='Gross Operating Profit',
        color='Network',
        barmode='group',
        title='Task 5: Gross Operating Profit by OTD Promise and Network',
        labels={'Gross Operating Profit': 'Gross Operating Profit ($)'},
        category_orders={"OTD Promise (Days)": ["1", "2", "3", "4", "5", "5+"]},
    )
    final_summary_df['Gross Operating Profit (M)'] = final_summary_df['Gross Operating Profit'] / 1e6
    fig.update_traces(
        text=final_summary_df['Gross Operating Profit (M)'].map(lambda x: f"{x:.2f}M"),
        texttemplate='<b>%{text}</b>',
        textposition='outside',
        textfont=dict(color='black', size=18)
    )
    chart_png_path = OUTPUT_DIR / "task5_profit_by_otd_chart.png"
    try:
        fig.write_image(chart_png_path, width=1000, height=550)
        print(f"-> Main profit chart saved to: {chart_png_path}")
    except Exception:
        print("\n--- Could not save image file. Please install kaleido by running: pip install kaleido ---")

    # graph 2: total revenue by otd value and network type
    final_summary_df['Total Revenue (M)'] = final_summary_df['Total Revenue'] / 1e6
    fig_total_revenue = px.bar(
        final_summary_df,
        x='OTD Promise (Days)',
        y='Total Revenue',
        color='Network',
        barmode='group',
        title='Task 5: Total Revenue by OTD Promise and Network',
        labels={'Total Revenue': 'Total Revenue ($)'},
        category_orders={"OTD Promise (Days)": ["1", "2", "3", "4", "5", "5+"]},
    )
    fig_total_revenue.update_traces(
        text=final_summary_df['Total Revenue (M)'].map(lambda x: f"{x:.2f}M"),
        texttemplate='<b>%{text}</b>',
        textposition='outside',
        textfont=dict(color='black', size=18)
    )
    total_revenue_chart_path = OUTPUT_DIR / "task5_total_revenue_bar.png"
    try:
        fig_total_revenue.write_image(total_revenue_chart_path, width=1000, height=550)
        print(f"-> Total revenue chart saved to: {total_revenue_chart_path}")
    except Exception:
        print("\n--- Could not save image file. Please install kaleido by running: pip install kaleido ---")

    # graph 3: net revenue by otd value and network type
    final_summary_df['Net Revenue (M)'] = final_summary_df['Net Revenue'] / 1e6
    fig_net_revenue = px.bar(
        final_summary_df,
        x='OTD Promise (Days)',
        y='Net Revenue',
        color='Network',
        barmode='group',
        title='Task 5: Net Revenue by OTD Promise and Network',
        labels={'Net Revenue': 'Net Revenue ($)'},
        category_orders={"OTD Promise (Days)": ["1", "2", "3", "4", "5", "5+"]},
    )
    fig_net_revenue.update_traces(
        text=final_summary_df['Net Revenue (M)'].map(lambda x: f"{x:.2f}M"),
        texttemplate='<b>%{text}</b>',
        textposition='outside',
        textfont=dict(color='black', size=18)
    )
    net_revenue_chart_path = OUTPUT_DIR / "task5_net_revenue_bar.png"
    try:
        fig_net_revenue.write_image(net_revenue_chart_path, width=1000, height=550)
        print(f"-> Net revenue chart saved to: {net_revenue_chart_path}")
    except Exception:
        print("\n--- Could not save image file. Please install kaleido by running: pip install kaleido ---")

    # graph 4: total COGS by otd value and network type
    final_summary_df['Total COGS (M)'] = final_summary_df['Total COGS'] / 1e6
    fig_cogs = px.bar(
        final_summary_df,
        x='OTD Promise (Days)',
        y='Total COGS',
        color='Network',
        barmode='group',
        title='Task 5: Total COGS by OTD Promise and Network',
        labels={'Total COGS': 'Total COGS ($)'},
        category_orders={"OTD Promise (Days)": ["1", "2", "3", "4", "5", "5+"]},
    )
    fig_cogs.update_traces(
        text=final_summary_df['Total COGS (M)'].map(lambda x: f"{x:.2f}M"),
        texttemplate='<b>%{text}</b>',
        textposition='outside',
        textfont=dict(color='black', size=18)
    )
    cogs_chart_path = OUTPUT_DIR / "task5_cogs_bar.png"
    try:
        fig_cogs.write_image(cogs_chart_path, width=1000, height=550)
        print(f"-> COGS chart saved to: {cogs_chart_path}")
    except Exception:
        print("\n--- Could not save image file. Please install kaleido by running: pip install kaleido ---")

    # graph 5: total shipping cost by otd value and network type
    final_summary_df['Total Shipping Cost (M)'] = final_summary_df['Total Shipping Cost'] / 1e6
    fig_shipping = px.bar(
        final_summary_df,
        x='OTD Promise (Days)',
        y='Total Shipping Cost',
        color='Network',
        barmode='group',
        title='Task 5: Total Shipping Cost by OTD Promise and Network',
        labels={'Total Shipping Cost': 'Total Shipping Cost ($)'},
        category_orders={"OTD Promise (Days)": ["1", "2", "3", "4", "5", "5+"]},
    )
    fig_shipping.update_traces(
        text=final_summary_df['Total Shipping Cost (M)'].map(lambda x: f"{x:.2f}M"),
        texttemplate='<b>%{text}</b>',
        textposition='outside',
        textfont=dict(color='black', size=18)
    )
    shipping_chart_path = OUTPUT_DIR / "task5_shipping_cost_bar.png"
    try:
        fig_shipping.write_image(shipping_chart_path, width=1000, height=550)
        print(f"-> Shipping cost chart saved to: {shipping_chart_path}")
    except Exception:
        print("\n--- Could not save image file. Please install kaleido by running: pip install kaleido ---")

    # graph 6: line chart showing trade-off between revenue, costs, and net revenue
    fig_tradeoff = px.line(
        final_summary_df, x='OTD Promise (Days)', y=['Total Revenue', 'Total Shipping Cost', 'Net Revenue'],
        facet_row='Network', title='Task 5: The Trade-Off Between Revenue, Costs, and Net Revenue',
        labels={'value': 'Amount ($)', 'variable': 'Metric'}, markers=True
    )
    tradeoff_chart_path = OUTPUT_DIR / "task5_tradeoff_chart.png"
    fig_tradeoff.write_image(tradeoff_chart_path, width=1000, height=800)
    print(f"-> Trade-off chart saved to: {tradeoff_chart_path}")


    print("\nScript complete.")
if __name__ == "__main__":
    main()