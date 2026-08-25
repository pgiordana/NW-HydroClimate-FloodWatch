#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json, math, sys, zipfile, shutil
import numpy as np
import pandas as pd
import xarray as xr

try:
    from shapely.geometry import shape, box
except Exception:
    print("ERRORE: manca shapely. Esegui: pip install shapely")
    sys.exit(2)

ROOT = Path(__file__).resolve().parent
ERA_ROOT = ROOT / "era5" / "event_2000_10_nw_daily"
P_DIR = ERA_ROOT / "pressure"
S_DIR = ERA_ROOT / "single"
RECEPTORS_FILE = ROOT / "basins_final" / "nw_receptors_final.geojson"

OUT = ROOT / "era5_analysis" / "event_2000_10_nw"
OUT.mkdir(parents=True, exist_ok=True)

BASIN_HOURLY = OUT / "basin_hourly_20001010_18.csv"
BASIN_SUMMARY = OUT / "basin_event_summary_20001010_18.csv"
DOMAIN_HOURLY = OUT / "domain_hourly_20001010_18.csv"
IVT_NC = OUT / "ivt_fields_20001010_18.nc"
QC = OUT / "era5_event_2000_qc.txt"

G = 9.80665

def pick_var(ds, aliases, required=True):
    for name in aliases:
        if name in ds.data_vars:
            return ds[name]
    low = {k.lower(): k for k in ds.data_vars}
    for name in aliases:
        if name.lower() in low:
            return ds[low[name.lower()]]
    if required:
        raise KeyError(f"Nessuna variabile fra {aliases}. Disponibili: {list(ds.data_vars)}")
    return None

def normalize(ds):
    rename = {}
    if "time" not in ds.dims and "time" not in ds.coords:
        for c in ("valid_time", "forecast_time"):
            if c in ds.dims or c in ds.coords:
                rename[c] = "time"; break
    if "pressure_level" not in ds.dims and "pressure_level" not in ds.coords:
        for c in ("level", "isobaricInhPa", "plev"):
            if c in ds.dims or c in ds.coords:
                rename[c] = "pressure_level"; break
    if "latitude" not in ds.dims and "latitude" not in ds.coords:
        for c in ("lat","y"):
            if c in ds.dims or c in ds.coords:
                rename[c] = "latitude"; break
    if "longitude" not in ds.dims and "longitude" not in ds.coords:
        for c in ("lon","x"):
            if c in ds.dims or c in ds.coords:
                rename[c] = "longitude"; break
    if rename:
        ds = ds.rename(rename)
    core = {"time","pressure_level","latitude","longitude"}
    for dim in list(ds.dims):
        if dim not in core and ds.sizes.get(dim,0) == 1:
            ds = ds.squeeze(dim, drop=True)
    for c in ("time","pressure_level","latitude","longitude"):
        if c in ds.coords:
            ds = ds.sortby(c)
    return ds

def open_concat(paths):
    dsets = [normalize(xr.open_dataset(p)) for p in paths]
    if not dsets:
        raise FileNotFoundError("Nessun file trovato.")
    ds = xr.concat(dsets, dim="time", data_vars="minimal",
                   coords="minimal", compat="override", join="override").sortby("time")
    _, idx = np.unique(pd.to_datetime(ds.time.values), return_index=True)
    return ds.isel(time=np.sort(idx))

def open_single_archive(path: Path, unpack_root: Path):
    """
    Dal novembre 2024 il convertitore CDS può restituire un archivio ZIP anche
    quando il target termina in .nc, se la richiesta contiene più stepType
    (per esempio variabili instantaneous + accumulated).
    Questa funzione rileva automaticamente il caso, estrae i NetCDF e li unisce.
    """
    if not zipfile.is_zipfile(path):
        return normalize(xr.open_dataset(path))

    day = path.stem.replace("era5_single_", "")
    dest = unpack_root / day
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".nc")]
        if not members:
            raise RuntimeError(f"{path.name} è ZIP ma non contiene file NetCDF.")
        for member in members:
            target = dest / Path(member).name
            if not target.exists() or target.stat().st_size == 0:
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    parts = []
    for ncfile in sorted(dest.glob("*.nc")):
        parts.append(normalize(xr.open_dataset(ncfile)))

    if not parts:
        raise RuntimeError(f"Nessun NetCDF estratto da {path.name}")

    # Instantaneous e accumulated hanno variabili diverse ma la stessa griglia/valid_time.
    # merge mantiene entrambe le famiglie nello stesso Dataset giornaliero.
    day_ds = xr.merge(parts, compat="override", join="outer")
    return normalize(day_ds)

def open_single_concat(paths, unpack_root):
    dsets = [open_single_archive(Path(p), unpack_root) for p in paths]
    if not dsets:
        raise FileNotFoundError("Nessun file single-level trovato.")
    ds = xr.concat(dsets, dim="time", data_vars="minimal",
                   coords="minimal", compat="override", join="outer").sortby("time")
    _, idx = np.unique(pd.to_datetime(ds.time.values), return_index=True)
    return ds.isel(time=np.sort(idx))

def met_from_dir(u, v):
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

def sector(deg):
    if not np.isfinite(deg):
        return "NA"
    labels = [("N","Tramontana"),("NE","Grecale"),("E","Levante"),("SE","Scirocco"),
              ("S","Ostro"),("SW","Libeccio"),("W","Ponente"),("NW","Maestrale")]
    code,name = labels[int(((deg+22.5)%360)//45)]
    return f"{code}-{name}"

def ivt_integral(q,u,v,levels_hpa,sp_pa):
    p = np.asarray(levels_hpa,float)*100.0
    order = np.argsort(p)
    p = p[order]; q=q[:,order]; u=u[:,order]; v=v[:,order]
    fu=q*u; fv=q*v
    valid = np.isfinite(fu)&np.isfinite(fv)&(p[None,:,None,None] <= sp_pa[:,None,:,:])
    nt,nz,ny,nx=q.shape
    iu=np.zeros((nt,ny,nx)); iv=np.zeros((nt,ny,nx))
    for k in range(nz-1):
        both=valid[:,k]&valid[:,k+1]
        dp=p[k+1]-p[k]
        iu += np.where(both,0.5*(fu[:,k]+fu[:,k+1])*dp,0.0)
        iv += np.where(both,0.5*(fv[:,k]+fv[:,k+1])*dp,0.0)
    for k in range(nz):
        last=valid[:,k].copy()
        if k<nz-1:
            last &= ~valid[:,k+1]
        gap=np.clip(sp_pa-p[k],0.0,None)
        iu += np.where(last,fu[:,k]*gap,0.0)
        iv += np.where(last,fv[:,k]*gap,0.0)
    anyv=np.any(valid,axis=1)
    iu=np.where(anyv,iu/G,np.nan); iv=np.where(anyv,iv/G,np.nan)
    mag=np.sqrt(iu**2+iv**2)
    return iu,iv,mag,met_from_dir(iu,iv)

def edges(vals):
    vals=np.asarray(vals,float)
    mids=(vals[:-1]+vals[1:])/2
    out=np.empty(len(vals)+1)
    out[1:-1]=mids
    out[0]=vals[0]-(mids[0]-vals[0])
    out[-1]=vals[-1]+(vals[-1]-mids[-1])
    return out

def receptor_weights(geom,lat,lon):
    le=edges(lat); oe=edges(lon)
    w=np.zeros((len(lat),len(lon)))
    minx,miny,maxx,maxy=geom.bounds
    for iy,y in enumerate(lat):
        y0,y1=sorted((le[iy],le[iy+1]))
        if y1<miny or y0>maxy: continue
        for ix,x in enumerate(lon):
            x0,x1=sorted((oe[ix],oe[ix+1]))
            if x1<minx or x0>maxx: continue
            inter=geom.intersection(box(x0,y0,x1,y1))
            if not inter.is_empty:
                w[iy,ix]=inter.area*math.cos(math.radians(float(y)))
    if w.sum()<=0:
        rp=geom.representative_point()
        iy=int(np.argmin(np.abs(lat-rp.y))); ix=int(np.argmin(np.abs(lon-rp.x)))
        w[iy,ix]=1.0
        return w,True
    return w,False

def wmean(arr,w):
    a=np.asarray(arr,float); valid=np.isfinite(a)
    num=np.nansum(np.where(valid,a*w[None],0.0),axis=(1,2))
    den=np.sum(np.where(valid,w[None],0.0),axis=(1,2))
    return np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)

def sel_level(da,lev):
    levels=np.asarray(da.pressure_level.values,float)
    i=int(np.argmin(np.abs(levels-lev)))
    if abs(levels[i]-lev)>1:
        raise ValueError(f"Livello {lev} hPa non trovato.")
    return da.isel(pressure_level=i)

def main():
    pfiles=sorted(P_DIR.glob("era5_pressure_*.nc"))
    sfiles=sorted(S_DIR.glob("era5_single_*.nc"))
    if len(pfiles)!=9 or len(sfiles)!=9:
        raise RuntimeError(f"Attesi 9+9 file; trovati {len(pfiles)}+{len(sfiles)}")
    if not RECEPTORS_FILE.exists():
        raise FileNotFoundError(RECEPTORS_FILE)

    print("Apro e concateno localmente i 18 file ERA5...")
    pds = open_concat(pfiles)

    unpack_root = ERA_ROOT / "single_unpacked"
    zip_count = sum(1 for p in sfiles if zipfile.is_zipfile(p))
    print(f"Single-level: {zip_count}/{len(sfiles)} file sono archivi ZIP CDS con estensione .nc.")
    sds = open_single_concat(sfiles, unpack_root)
    print("Pressure variables:",list(pds.data_vars))
    print("Single variables:",list(sds.data_vars))

    u_da=pick_var(pds,["u","u_component_of_wind"])
    v_da=pick_var(pds,["v","v_component_of_wind"])
    q_da=pick_var(pds,["q","specific_humidity"])
    t_da=pick_var(pds,["t","temperature"])

    # Allinea single a time/grid pressure.
    sds=sds.reindex(time=pds.time.values,method="nearest")
    sds=sds.reindex(latitude=pds.latitude.values,longitude=pds.longitude.values,method="nearest")

    sp_da=pick_var(sds,["sp","surface_pressure"])
    u10_da=pick_var(sds,["u10","10m_u_component_of_wind"])
    v10_da=pick_var(sds,["v10","10m_v_component_of_wind"])
    msl_da=pick_var(sds,["msl","mean_sea_level_pressure"])
    tcwv_da=pick_var(sds,["tcwv","total_column_water_vapour"])
    cape_da=pick_var(sds,["cape","convective_available_potential_energy"])
    tp_da=pick_var(sds,["tp","total_precipitation"])
    e_da=pick_var(sds,["e","evaporation"])
    sw1_da=pick_var(sds,["swvl1","volumetric_soil_water_layer_1"])
    sw2_da=pick_var(sds,["swvl2","volumetric_soil_water_layer_2"])
    sw3_da=pick_var(sds,["swvl3","volumetric_soil_water_layer_3"])

    u=np.asarray(u_da.transpose("time","pressure_level","latitude","longitude").values,float)
    v=np.asarray(v_da.transpose("time","pressure_level","latitude","longitude").values,float)
    q=np.asarray(q_da.transpose("time","pressure_level","latitude","longitude").values,float)
    sp=np.asarray(sp_da.transpose("time","latitude","longitude").values,float)
    levels=np.asarray(pds.pressure_level.values,float)
    lat=np.asarray(pds.latitude.values,float); lon=np.asarray(pds.longitude.values,float)
    times=pd.to_datetime(pds.time.values)

    print("Calcolo IVT 300 hPa -> superficie locale...")
    ivtx,ivty,ivt,ivtfrom=ivt_integral(q,u,v,levels,sp)

    xr.Dataset(
        {
            "ivtx_300_surface":(("time","latitude","longitude"),ivtx.astype("f4")),
            "ivty_300_surface":(("time","latitude","longitude"),ivty.astype("f4")),
            "ivt_300_surface":(("time","latitude","longitude"),ivt.astype("f4")),
            "ivt_from_direction":(("time","latitude","longitude"),ivtfrom.astype("f4")),
        },
        coords={"time":pds.time.values,"latitude":lat,"longitude":lon},
        attrs={
            "event":"NW Italy 10-18 October 2000",
            "ivt_definition":"1/g integral q*(u,v) dp from 300 hPa to local surface",
            "below_ground_handling":"p > surface pressure excluded; surface gap approximated with highest valid level"
        }
    ).to_netcdf(IVT_NC)

    domain=[]
    for i,ts in enumerate(times):
        a=ivt[i]
        if np.isfinite(a).any():
            iy,ix=np.unravel_index(np.nanargmax(a),a.shape)
            d=float(ivtfrom[i,iy,ix])
            domain.append({"time":ts.isoformat(),"domain_max_ivt_kg_m_s":float(a[iy,ix]),
                           "domain_max_ivt_lat":float(lat[iy]),"domain_max_ivt_lon":float(lon[ix]),
                           "domain_max_ivt_from_deg":d,"domain_max_ivt_sector":sector(d)})
    pd.DataFrame(domain).to_csv(DOMAIN_HOURLY,index=False)

    fc=json.loads(RECEPTORS_FILE.read_text(encoding="utf-8"))
    receptors=[]
    for feat in fc["features"]:
        p=feat["properties"]
        receptors.append({"receptor_id":p["receptor_id"],"label":p["label"],
                          "region":p["region"],"priority":p["priority"],
                          "geometry":shape(feat["geometry"])})

    print(f"Costruisco pesi di sovrapposizione per {len(receptors)} recettori...")
    weights={}; fallback={}
    for r in receptors:
        w,fb=receptor_weights(r["geometry"],lat,lon)
        weights[r["receptor_id"]]=w; fallback[r["receptor_id"]]=fb
        print(f"  {r['receptor_id']}: celle intersecate={int(np.sum(w>0))}, fallback={fb}")

    plev={}
    for lev in (925,850,700):
        plev[lev]={}
        for key,da in (("u",u_da),("v",v_da),("q",q_da),("t",t_da)):
            plev[lev][key]=np.asarray(sel_level(da,lev).transpose("time","latitude","longitude").values,float)

    singles={
        "u10":np.asarray(u10_da.transpose("time","latitude","longitude").values,float),
        "v10":np.asarray(v10_da.transpose("time","latitude","longitude").values,float),
        "msl":np.asarray(msl_da.transpose("time","latitude","longitude").values,float),
        "sp":np.asarray(sp_da.transpose("time","latitude","longitude").values,float),
        "tcwv":np.asarray(tcwv_da.transpose("time","latitude","longitude").values,float),
        "cape":np.asarray(cape_da.transpose("time","latitude","longitude").values,float),
        "tp":np.asarray(tp_da.transpose("time","latitude","longitude").values,float),
        "e":np.asarray(e_da.transpose("time","latitude","longitude").values,float),
        "sw1":np.asarray(sw1_da.transpose("time","latitude","longitude").values,float),
        "sw2":np.asarray(sw2_da.transpose("time","latitude","longitude").values,float),
        "sw3":np.asarray(sw3_da.transpose("time","latitude","longitude").values,float),
    }

    rows=[]
    for r in receptors:
        rid=r["receptor_id"]; w=weights[rid]
        bx=wmean(ivtx,w); by=wmean(ivty,w); bmag=np.sqrt(bx**2+by**2); bfrom=met_from_dir(bx,by)
        sv={k:wmean(vv,w) for k,vv in singles.items()}
        pv={}
        for lev,d in plev.items():
            pu=wmean(d["u"],w); pvv=wmean(d["v"],w)
            pv[lev]={"u":pu,"v":pvv,"speed":np.sqrt(pu**2+pvv**2),
                     "from":met_from_dir(pu,pvv),"q":wmean(d["q"],w),"t":wmean(d["t"],w)}
        for i,ts in enumerate(times):
            row={
                "time":ts.isoformat(),"receptor_id":rid,"label":r["label"],
                "region":r["region"],"priority":r["priority"],
                "era5_cells_intersected":int(np.sum(w>0)),
                "era5_nearest_cell_fallback":bool(fallback[rid]),
                "ivtx_kg_m_s":bx[i],"ivty_kg_m_s":by[i],"ivt_kg_m_s":bmag[i],
                "ivt_from_deg":bfrom[i],"ivt_sector":sector(bfrom[i]),
                "u10_m_s":sv["u10"][i],"v10_m_s":sv["v10"][i],
                "wind10_speed_m_s":math.hypot(sv["u10"][i],sv["v10"][i]),
                "wind10_from_deg":met_from_dir(sv["u10"][i],sv["v10"][i]),
                "mslp_hpa":sv["msl"][i]/100.0,"surface_pressure_hpa":sv["sp"][i]/100.0,
                "tcwv_kg_m2":sv["tcwv"][i],"cape_J_kg":sv["cape"][i],
                "precip_hour_mm":sv["tp"][i]*1000.0,
                "evap_raw_mm":sv["e"][i]*1000.0,
                "evap_upward_positive_mm":-sv["e"][i]*1000.0,
                "soil_water_l1_m3_m3":sv["sw1"][i],"soil_water_l2_m3_m3":sv["sw2"][i],
                "soil_water_l3_m3_m3":sv["sw3"][i],
            }
            for lev in (925,850,700):
                row.update({
                    f"u{lev}_m_s":pv[lev]["u"][i],f"v{lev}_m_s":pv[lev]["v"][i],
                    f"wind{lev}_speed_m_s":pv[lev]["speed"][i],
                    f"wind{lev}_from_deg":pv[lev]["from"][i],
                    f"wind{lev}_sector":sector(pv[lev]["from"][i]),
                    f"q{lev}_kg_kg":pv[lev]["q"][i],f"t{lev}_K":pv[lev]["t"][i],
                })
            rows.append(row)

    df=pd.DataFrame(rows)
    df["time"]=pd.to_datetime(df["time"])
    parts=[]
    for rid,gdf in df.groupby("receptor_id",sort=False):
        gdf=gdf.sort_values("time").copy()
        for h in (6,12,24,48):
            gdf[f"precip_{h}h_mm"]=gdf["precip_hour_mm"].rolling(h,min_periods=1).sum()
        run=[]; prev=None; n=0
        for s in gdf["ivt_sector"]:
            if s==prev: n+=1
            else: prev=s; n=1
            run.append(n)
        gdf["ivt_sector_persistence_h"]=run
        parts.append(gdf)
    df=pd.concat(parts,ignore_index=True)
    df.to_csv(BASIN_HOURLY,index=False)

    summaries=[]
    for rid,gdf in df.groupby("receptor_id",sort=False):
        gdf=gdf.sort_values("time")
        i_ivt=gdf["ivt_kg_m_s"].idxmax()
        i24=gdf["precip_24h_mm"].idxmax()
        i48=gdf["precip_48h_mm"].idxmax()
        i_cape=gdf["cape_J_kg"].idxmax()
        summaries.append({
            "receptor_id":rid,"label":gdf["label"].iloc[0],"region":gdf["region"].iloc[0],
            "priority":gdf["priority"].iloc[0],
            "era5_cells_intersected":int(gdf["era5_cells_intersected"].iloc[0]),
            "nearest_cell_fallback":bool(gdf["era5_nearest_cell_fallback"].iloc[0]),
            "max_ivt_kg_m_s":df.loc[i_ivt,"ivt_kg_m_s"],
            "max_ivt_time":df.loc[i_ivt,"time"].isoformat(),
            "max_ivt_from_deg":df.loc[i_ivt,"ivt_from_deg"],
            "max_ivt_sector":df.loc[i_ivt,"ivt_sector"],
            "max_wind850_m_s":gdf["wind850_speed_m_s"].max(),
            "dominant_ivt_sector":gdf["ivt_sector"].mode().iloc[0],
            "max_sector_persistence_h":int(gdf["ivt_sector_persistence_h"].max()),
            "retrieved_period_precip_mm":gdf["precip_hour_mm"].sum(),
            "max_24h_precip_mm":df.loc[i24,"precip_24h_mm"],
            "max_24h_precip_end_time":df.loc[i24,"time"].isoformat(),
            "max_48h_precip_mm":df.loc[i48,"precip_48h_mm"],
            "max_48h_precip_end_time":df.loc[i48,"time"].isoformat(),
            "max_cape_J_kg":df.loc[i_cape,"cape_J_kg"],
            "max_cape_time":df.loc[i_cape,"time"].isoformat(),
            "mean_tcwv_kg_m2":gdf["tcwv_kg_m2"].mean(),
            "max_tcwv_kg_m2":gdf["tcwv_kg_m2"].max(),
            "mean_soil_water_l1":gdf["soil_water_l1_m3_m3"].mean(),
            "mean_soil_water_l2":gdf["soil_water_l2_m3_m3"].mean(),
            "mean_soil_water_l3":gdf["soil_water_l3_m3_m3"].mean(),
        })
    sdf=pd.DataFrame(summaries).sort_values(["max_48h_precip_mm","max_ivt_kg_m_s"],ascending=False)
    sdf.to_csv(BASIN_SUMMARY,index=False)

    qc=[
        "ERA5 EVENT 2000 — QC",
        f"Pressure files: {len(pfiles)}",f"Single files: {len(sfiles)}",
        f"Single ZIP archives detected: {zip_count}",
        f"Pressure hours: {pds.sizes['time']}",f"Single hours: {sds.sizes['time']}",
        "Expected hours: 216",f"Pressure levels: {list(np.asarray(pds.pressure_level.values,float))}",
        f"Grid: {len(lat)} x {len(lon)}",f"Time start: {times.min()}",
        f"Time end: {times.max()}",f"Receptors: {len(receptors)}",
        f"Nearest-cell fallback receptors: {sum(fallback.values())}",
        "",
        "IVT method:",
        "- partial-column 300 hPa to local surface",
        "- below-ground pressure levels removed using ERA5 surface pressure",
        "- surface gap approximated using highest valid pressure-level moisture flux",
        "",
        "Precipitation:",
        "- tp converted from m to mm",
        "- hourly ERA5 tp interpreted as accumulation over the hour ending at timestamp",
        "- rolling 6/12/24/48 h sums computed from hourly values",
        "",
        "IMPORTANT:",
        "- This is a diagnostic event reconstruction, not yet a probabilistic flood model.",
        "- Basin averages use fractional overlap between ERA5 grid cells and receptor polygons.",
        "- Small Ligurian basins may still be poorly resolved at 0.25-degree ERA5 scale."
    ]
    low=sdf[sdf["era5_cells_intersected"]<3]
    if not low.empty:
        qc+=["","LOW-RESOLUTION RECEPTORS (<3 intersected ERA5 cells):"]
        for _,rr in low.iterrows():
            qc.append(f"- {rr['receptor_id']} ({rr['label']}): {int(rr['era5_cells_intersected'])} cells")
    QC.write_text("\n".join(qc),encoding="utf-8")

    print("\nANALISI ERA5 EVENTO 2000 COMPLETATA.")
    for p in (BASIN_HOURLY,BASIN_SUMMARY,DOMAIN_HOURLY,IVT_NC,QC):
        print(" ",p)
    print("\nGraduatoria preliminare per massima precipitazione 48h:")
    print(sdf[["receptor_id","max_48h_precip_mm","max_ivt_kg_m_s","max_ivt_sector","max_wind850_m_s"]].head(10).to_string(index=False))

if __name__=="__main__":
    main()
