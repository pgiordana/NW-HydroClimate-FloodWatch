#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

START_YEAR, END_YEAR = 1987, 2025
MONTHS = [9,10,11,12]
FAMILIES = {
    "source3h": {
        "subdir":"source_single_3h",
        "pattern":"era5_source_single_3h_{ym}.nc",
        "freq_h":3,
        "vars":{
            "ivt_e":["vertical_integral_of_eastward_water_vapour_flux","viwve","ivt_e","ivte"],
            "ivt_n":["vertical_integral_of_northward_water_vapour_flux","viwvn","ivt_n","ivtn"],
            "tcwv":["total_column_water_vapour","tcwv"],
            "cape":["convective_available_potential_energy","cape"],
            "mslp":["mean_sea_level_pressure","msl","mslp"],
        },
    },
    "pressure3h": {
        "subdir":"pressure_3h",
        "pattern":"era5_pressure_3h_{ym}.nc",
        "freq_h":3,
        "vars":{
            "u":["u","u_component_of_wind"],
            "v":["v","v_component_of_wind"],
            "q":["q","specific_humidity"],
            "t":["t","temperature"],
        },
        "levels":[925,850,700],
    },
    "precip1h": {
        "subdir":"target_precip_hourly",
        "pattern":"era5_target_precip_1h_{ym}.nc",
        "freq_h":1,
        "vars":{"tp":["tp","total_precipitation"]},
    },
    "state1d": {
        "subdir":"target_state_daily",
        "pattern":"era5_target_state_1d_{ym}.nc",
        "freq_h":24,
        "vars":{
            "swvl1":["swvl1","volumetric_soil_water_layer_1"],
            "swvl2":["swvl2","volumetric_soil_water_layer_2"],
            "swvl3":["swvl3","volumetric_soil_water_layer_3"],
            "sd":["sd","snow_depth"],
        },
    },
}

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def expected_times(y,m,freq):
    s=pd.Timestamp(y,m,1)
    e=pd.Timestamp(y+1,1,1) if m==12 else pd.Timestamp(y,m+1,1)
    return pd.date_range(s,e,freq=f"{freq}h",inclusive="left")

def find_time(ds):
    for c in ("valid_time","time","forecast_time"):
        if c in ds.coords or c in ds.variables: return c
    for c in ds.coords:
        try:
            if np.issubdtype(ds[c].dtype,np.datetime64): return c
        except: pass
    return None

def find_coord(ds,names):
    for c in names:
        if c in ds.coords or c in ds.variables: return c
    return None

def find_level(ds):
    for c in ("pressure_level","level","isobaricInhPa","isobaricInPa","plev"):
        if c in ds.coords or c in ds.variables: return c
    return None

def resolve_var(ds,aliases):
    names=set(ds.data_vars)|set(ds.variables)
    for a in aliases:
        if a in names:return a
    low={str(n).lower():n for n in names}
    for a in aliases:
        if a.lower() in low:return low[a.lower()]
    return None

def grid_sig(ds):
    la=find_coord(ds,("latitude","lat")); lo=find_coord(ds,("longitude","lon"))
    if not la or not lo:return None
    def one(a):
        v=np.asarray(a).ravel()
        return (tuple(np.asarray(a).shape),float(np.nanmin(v)),float(np.nanmax(v)),float(v[0]),float(v[-1]))
    return {"lat":one(ds[la].values),"lon":one(ds[lo].values)}

def same_grid(a,b):
    if a is None or b is None:return False
    if a["lat"][0]!=b["lat"][0] or a["lon"][0]!=b["lon"][0]:return False
    for ax in ("lat","lon"):
        for i in range(1,5):
            if not math.isclose(a[ax][i],b[ax][i],abs_tol=1e-8,rel_tol=0):return False
    return True

def audit_one(path,fam,y,m):
    cfg=FAMILIES[fam]
    r={"family":fam,"year":y,"month":m,"path":str(path),"exists":path.exists(),
       "status":"MISSING","error":"","size_bytes":None,"sha256":None,
       "time_count":None,"expected_time_count":None,"time_min":"","time_max":"",
       "missing_times":None,"extra_times":None,"duplicate_times":None,
       "out_of_order_times":None,"out_of_month_times":None,"median_step_hours":None,
       "vars_missing":"","levels":"","levels_ok":None,"grid_ok":None}
    if not path.exists(): return r,None
    r["size_bytes"]=path.stat().st_size
    if r["size_bytes"]<=0:
        r["status"]="FAIL";r["error"]="zero_byte";return r,None
    try:r["sha256"]=sha256_file(path)
    except Exception as e:
        r["status"]="FAIL";r["error"]=f"sha256:{e!r}";return r,None
    try:
        with xr.open_dataset(path,decode_times=True) as ds:
            tn=find_time(ds)
            if not tn:
                r["status"]="FAIL";r["error"]="time_not_found";return r,grid_sig(ds)
            t=pd.DatetimeIndex(pd.to_datetime(np.asarray(ds[tn].values).ravel(),errors="coerce"))
            exp=expected_times(y,m,cfg["freq_h"])
            valid=t[~pd.isna(t)]
            r["time_count"]=len(t);r["expected_time_count"]=len(exp)
            if len(valid):
                r["time_min"]=str(valid.min());r["time_max"]=str(valid.max())
            r["duplicate_times"]=int(valid.duplicated().sum())
            if len(valid)>1:
                hours=pd.Series(valid).diff().dropna().dt.total_seconds()/3600
                r["median_step_hours"]=float(hours.median())
                r["out_of_order_times"]=int((hours<0).sum())
            else:r["out_of_order_times"]=0
            uniq=pd.DatetimeIndex(valid.unique()).sort_values()
            r["missing_times"]=len(exp.difference(uniq))
            r["extra_times"]=len(uniq.difference(exp))
            r["out_of_month_times"]=int((~((valid.year==y)&(valid.month==m))).sum())
            miss=[]
            for logical,aliases in cfg["vars"].items():
                if resolve_var(ds,aliases) is None:miss.append(logical)
            r["vars_missing"]=";".join(miss)
            if fam=="pressure3h":
                ln=find_level(ds)
                if ln:
                    vals=pd.to_numeric(pd.Series(np.asarray(ds[ln].values).ravel()),errors="coerce").dropna().astype(float)
                    if len(vals) and vals.max()>2000: vals=vals/100
                    levels=sorted(set(int(round(v)) for v in vals))
                    r["levels"]=",".join(map(str,levels))
                    r["levels_ok"]=all(x in levels for x in cfg["levels"])
                else:r["levels_ok"]=False
            fatal=[]
            if miss:fatal.append("missing_vars")
            if r["time_count"]!=r["expected_time_count"]:fatal.append("time_count")
            if r["duplicate_times"]:fatal.append("duplicates")
            if r["out_of_order_times"]:fatal.append("out_of_order")
            if r["missing_times"]:fatal.append("missing_times")
            if r["extra_times"]:fatal.append("extra_times")
            if r["out_of_month_times"]:fatal.append("out_of_month")
            if fam=="pressure3h" and not r["levels_ok"]:fatal.append("levels")
            gs=grid_sig(ds)
            if gs is None:fatal.append("grid")
            r["status"]="FAIL" if fatal else "PASS"
            r["error"]=";".join(fatal)
            return r,gs
    except Exception as e:
        r["status"]="ERROR";r["error"]=repr(e);return r,None

def main():
    root=Path(__file__).resolve().parent
    era=root/"era5_historical_nw"
    out=era/"audit"/"era5_v1_0"
    out.mkdir(parents=True,exist_ok=True)
    print("="*110)
    print("ERA5 HISTORICAL NW — AUDIT DEFINITIVO v1.0")
    print("Attesi: 156 mesi x 4 famiglie = 624 NetCDF")
    print("="*110)
    rows=[]; refs={}
    total=624;done=0
    for y in range(START_YEAR,END_YEAR+1):
        for m in MONTHS:
            ym=f"{y}{m:02d}"
            for fam,cfg in FAMILIES.items():
                p=era/cfg["subdir"]/str(y)/cfg["pattern"].format(ym=ym)
                r,g=audit_one(p,fam,y,m)
                if g is not None:
                    if fam not in refs:refs[fam]=g;r["grid_ok"]=True
                    else:
                        r["grid_ok"]=same_grid(refs[fam],g)
                        if not r["grid_ok"] and r["status"]=="PASS":
                            r["status"]="FAIL";r["error"]="grid_mismatch"
                rows.append(r);done+=1
                if r["status"]!="PASS":
                    print(f"{y}-{m:02d} | {fam:<11} | {r['status']:<7} | {r['error']}")
                if done%32==0 or done==total:print(f"PROGRESSO AUDIT: {done}/{total}")
    df=pd.DataFrame(rows)
    counts=Counter(df["status"])
    overall="PASS" if counts["PASS"]==624 else "REVIEW"
    df.to_csv(out/"era5_file_audit_v1_0.csv",index=False)
    report={"version":"1.0","overall_status":overall,"expected_files":624,
            "observed_files":int(df["exists"].sum()),"status_counts":dict(counts),
            "raw_modified":False}
    (out/"era5_audit_v1_0.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["="*110,"ERA5 HISTORICAL NW — AUDIT DEFINITIVO v1.0","="*110,
           f"OVERALL STATUS : {overall}",f"File attesi    : 624",
           f"File presenti  : {int(df['exists'].sum())}",f"PASS           : {counts['PASS']}",
           f"FAIL           : {counts['FAIL']}",f"ERROR          : {counts['ERROR']}",
           f"MISSING        : {counts['MISSING']}",f"Output         : {out}"]
    (out/"era5_audit_v1_0.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n"+"="*110)
    for x in lines[3:]:print(x)
    print("="*110)
    if overall!="PASS":
        bad=df[df["status"]!="PASS"][["family","year","month","status","error","path"]]
        if len(bad):
            print("\nFILE DA RIVEDERE:")
            print(bad.to_string(index=False))
        sys.exit(2)

if __name__=="__main__":
    main()
