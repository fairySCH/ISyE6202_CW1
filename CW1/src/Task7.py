import os, numpy as np, pandas as pd

# ---- CONFIG ----
P = {
    "zip3_pmf": "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/data/zip3_pmf.csv",
    "fc_asg":   "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task3/assignment_15FC.csv",
    "daily":    "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task1_2/df_global2_seasonality_daily.csv",
    "val_col":  "units_2026_expected_distnormal_99",
}
OUT = "/Users/shankaraadhithyaa/Desktop/Python/ISyE6202_CW1/CW1/output/task7"
YR, Z_FC, Z_NET, CV = 2026, 2.33, 2.33, 0.15
FC_DAYS, NET_DAYS = 21, 56
DATES = pd.date_range(f"{YR}-01-01", f"{YR}-12-31", freq="D")
SCEN_Z, SCEN_W = {"50%":0.0,"68%":1.0,"95%":1.645}, [6,12]

# ---- HELPERS ----
def must(p): 
    if not os.path.exists(p): raise FileNotFoundError(p); 
    return p
def norm(df): 
    df.columns=[c.strip().lower() for c in df.columns]; return df
def inv_window(a, cv, z):
    a=np.asarray(a,float)
    if a.size==0: return 0.0,0.0,0.0
    cyc=a.sum(); saf=z*np.sqrt(np.sum((cv*a)**2))
    return cyc, saf, cyc+saf

# ---- LOAD ----
def load_zip3(p):
    df=norm(pd.read_csv(must(p),dtype=str)); df["zip3"]=df["zip3"].astype(str).str.zfill(3); df["pmf"]=df["pmf"].astype(float)
    s=df["pmf"].sum()
    if s>0 and not np.isclose(s,1.0): df["pmf"]/=s
    return df[["zip3","pmf"]]

def load_asg(p):
    df=norm(pd.read_csv(must(p),dtype=str))
    if "preferred_fc" in df.columns and "fc_id" not in df.columns: df=df.rename(columns={"preferred_fc":"fc_id"})
    if "zip3" not in df.columns or "fc_id" not in df.columns: raise KeyError("Need columns: zip3, fc_id")
    df["zip3"]=df["zip3"].astype(str).str.zfill(3)
    return df[["zip3","fc_id"]]

def load_daily(p, dates, col):
    df=norm(pd.read_csv(must(p)))
    if "date" not in df.columns or col not in df.columns: raise KeyError("daily needs 'date' and value col")
    s=pd.Series(df[col].astype(float).values, index=pd.to_datetime(df["date"]))
    return pd.Series(s.reindex(dates).fillna(0.0).values, index=dates, name="units")

def mk_zip3_daily(units, zdf):  
    return pd.DataFrame({r.zip3: units.values*r.pmf for _,r in zdf.iterrows()}, index=units.index)

def zip3_to_fc(zdf, asg):
    mp=asg.set_index("zip3")["fc_id"].to_dict()
    fc_cols={}
    for z in zdf.columns:
        fc=mp.get(z)
        if fc: fc_cols.setdefault(fc,[]).append(z)
    return pd.DataFrame({fc:zdf[c].sum(axis=1).values for fc,c in fc_cols.items()}, index=zdf.index).sort_index(axis=1)

# ---- MAIN ----
def main():
    os.makedirs(OUT, exist_ok=True)
    zdf, asg, units = load_zip3(P["zip3_pmf"]), load_asg(P["fc_asg"]), load_daily(P["daily"], DATES, P["val_col"])
    fc_d = zip3_to_fc(mk_zip3_daily(units, zdf), asg)
    if fc_d.shape[1]==0: raise RuntimeError("No FC columns—check assignments/PMF")

    # FC daily (3w @ 99%)
    fc_rows=[]
    for fc in fc_d.columns:
        a=fc_d[fc].values
        for i,dt in enumerate(fc_d.index):
            cyc,saf,tgt=inv_window(a[i:i+FC_DAYS], CV, Z_FC)
            fc_rows.append({"Date":dt.strftime("%Y-%m-%d"),"FC ID":fc,"FC Average":cyc,"FC Safety":saf,"FC Target":tgt})
    df_fc=pd.DataFrame(fc_rows)

    # Network + DC (8w @ 99%)
    net_arr=fc_d.sum(axis=1).values; net_rows=[]
    for i,dt in enumerate(fc_d.index):
        nc,ns,nt=inv_window(net_arr[i:i+NET_DAYS], CV, Z_NET)
        d=dt.strftime("%Y-%m-%d"); fc_sum=df_fc.loc[df_fc["Date"]==d,"FC Target"].sum()
        net_rows.append({"Date":d,"Network Average":nc,"Network Safety":ns,"Network Target":nt,"DC Inventory":max(0,nt-fc_sum)})
    df_net=pd.DataFrame(net_rows)

    # Summaries
    df_fc_max = df_fc.groupby("FC ID",as_index=False)["FC Target"].agg(**{"FC Target Max":"max","FC Target Mean":"mean"})
    df_fc_max_only = df_fc_max[["FC ID","FC Target Max"]]
    df_net_max = pd.DataFrame([{
        "Network Target Max":df_net["Network Target"].max(),
        "Network Target Mean":df_net["Network Target"].mean(),
        "DC Inventory Max":df_net["DC Inventory"].max(),
        "DC Inventory Mean":df_net["DC Inventory"].mean()
    }])

    # Scenarios (6/12w × 50/68/95)
    scen=[]
    for w in SCEN_W:
        nd=w*7
        for lvl,z in SCEN_Z.items():
            for i,dt in enumerate(fc_d.index):
                nc,ns,nt=inv_window(net_arr[i:i+nd], CV, z)
                d=dt.strftime("%Y-%m-%d"); fc_sum=df_fc.loc[df_fc["Date"]==d,"FC Target"].sum()
                scen.append({"Date":d,"Weeks":w,"Service Level":lvl,"Z":z,
                             "Network Average":nc,"Network Safety":ns,"Network Target":nt,
                             "FC Total Target Today":fc_sum,"DC Inventory":max(0,nt-fc_sum)})
    df_scen=pd.DataFrame(scen)
    scen_summary=(df_scen.groupby(["Weeks","Service Level"],as_index=False)
                  .agg(**{"Network Max Target":("Network Target","max"),
                          "Network Mean Target":("Network Target","mean"),
                          "DC Inventory Max":("DC Inventory","max"),
                          "DC Inventory Mean":("DC Inventory","mean")}))

    # Save CSVs
    df_fc.to_csv(os.path.join(OUT,"Task7 FC Daily.csv"),index=False)
    df_net.to_csv(os.path.join(OUT,"Task7 Network Daily.csv"),index=False)
    df_fc_max_only.to_csv(os.path.join(OUT,"Task7 FC Max.csv"),index=False)
    df_fc_max.to_csv(os.path.join(OUT,"Task7 FC Max & Mean.csv"),index=False)
    df_net_max.to_csv(os.path.join(OUT,"Task7 Network Max.csv"),index=False)
    df_scen.to_csv(os.path.join(OUT,"Task7 Scenarios Daily.csv"),index=False)
    scen_summary.to_csv(os.path.join(OUT,"Task7 Scenarios Summary.csv"),index=False)

    # Also save a single Excel combining all CSVs
    with pd.ExcelWriter(os.path.join(OUT,"Task7 Report.xlsx")) as xw:
        df_fc.to_excel(xw, sheet_name="FC Daily", index=False)
        df_net.to_excel(xw, sheet_name="Network Daily", index=False)
        df_fc_max_only.to_excel(xw, sheet_name="FC Max", index=False)
        df_fc_max.to_excel(xw, sheet_name="FC Max & Mean", index=False)
        df_net_max.to_excel(xw, sheet_name="Network Max", index=False)
        df_scen.to_excel(xw, sheet_name="Scenarios Daily", index=False)
        scen_summary.to_excel(xw, sheet_name="Scenarios Summary", index=False)

    print("✅ CSVs + Task7 Report.xlsx saved in", OUT)

if __name__=="__main__":
    main()
