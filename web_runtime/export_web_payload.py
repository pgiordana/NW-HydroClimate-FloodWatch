#!/usr/bin/env python3
import argparse,json,math,shutil
from datetime import datetime,timezone
from pathlib import Path

def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,list): return [clean(x) for x in v]
    if isinstance(v,float) and not math.isfinite(v): return None
    return v

def latest_run(root):
    out=root/'nw_floodwatch_output'
    runs=[p for p in out.iterdir() if p.is_dir() and (p/'NW_FloodWatch_predictions.json').exists() and (p/'NW_FloodWatch_run_audit.json').exists()]
    if not runs: raise RuntimeError('No completed NW FloodWatch output run found')
    return sorted(runs,key=lambda p:p.name)[-1]

def copy(src,dst):
    if src.exists(): dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runtime-root',required=True); ap.add_argument('--site-dir',required=True); ap.add_argument('--demo',action='store_true'); a=ap.parse_args()
    root=Path(a.runtime_root).resolve(); site=Path(a.site_dir).resolve(); (site/'data').mkdir(parents=True,exist_ok=True); (site/'downloads').mkdir(parents=True,exist_ok=True)
    geo=root/'nw_hydroclimate_core_release_v1_0'/'metadata'/'nw_receptors_final.geojson'; copy(geo,site/'data'/'receptors.geojson')
    if a.demo:
        ids=['LIG_BISAGNO','LIG_CENTA','LIG_MAGRA','LIG_POLCEVERA','NW_BORMIDA','NW_CHISONE','NW_DORA_BALTEA','NW_DORA_RIPARIA','NW_MAIRA','NW_ORBA','NW_ORCO','NW_PELLICE','NW_SCRIVIA','NW_SESIA','NW_STURA_DEMONTE','NW_STURA_LANZO','NW_TANARO_ALTO','NW_TANARO_MEDIO_BASSO','NW_TOCE','NW_VARAITA']
        copy(root/'NW_FloodWatch_Bollettino_DEMO.pdf',site/'downloads'/'NW_FloodWatch_Bollettino_DEMO.pdf')
        rec=[{'receptor_id':r,'label':r.replace('NW_','').replace('LIG_','').replace('_',' ').title(),'probability_24h':None,'probability_48h':None,'probability_72h':None,'overall_semaphore':'GRAY','action_note':'Demo tecnica: non interpretare.'} for r in ids]
        payload={'schema':'nwfloodwatch.web.latest.v1','product':'NW FloodWatch','experimental':True,'official_warning_use_allowed':False,'generated_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'run_id':'DEMO_LAYOUT_ONLY','issue_date':None,'bulletin_mode':'DEMO_LAYOUT_ONLY','scientific_beta_interpretation':False,'in_core_season':False,'model_inference_performed':False,'prospective_beta_allowed':False,'summary':['Demo tecnica del sito: nessuna previsione reale e nessun dato operativo da interpretare.'],'bulletin_pdf':'downloads/NW_FloodWatch_Bollettino_DEMO.pdf','receptors':rec}
    else:
        run=latest_run(root); preds=json.loads((run/'NW_FloodWatch_predictions.json').read_text()); audit=json.loads((run/'NW_FloodWatch_run_audit.json').read_text())
        copy(run/'NW_FloodWatch_Bollettino.pdf',site/'downloads'/'LATEST_NW_FloodWatch_Bollettino.pdf')
        rec=[{'receptor_id':x.get('receptor_id'),'label':x.get('basin_label'),'probability_24h':x.get('calibrated_probability_24h'),'probability_48h':x.get('calibrated_probability_48h'),'probability_72h':x.get('calibrated_probability_72h'),'overall_semaphore':x.get('overall_semaphore','GRAY'),'action_note':x.get('action_note','')} for x in preds]
        payload={'schema':'nwfloodwatch.web.latest.v1','product':'NW FloodWatch','experimental':True,'official_warning_use_allowed':False,'generated_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'run_id':audit.get('run_id',run.name),'issue_date':audit.get('issue_date'),'bulletin_mode':audit.get('bulletin_mode'),'scientific_beta_interpretation':audit.get('scientific_beta_interpretation',False),'in_core_season':audit.get('in_core_season',False),'model_inference_performed':audit.get('model_inference_performed',False),'prospective_beta_allowed':audit.get('prospective_beta_allowed',False),'summary':audit.get('summary',[]),'bulletin_pdf':'downloads/LATEST_NW_FloodWatch_Bollettino.pdf','receptors':rec}
    (site/'data'/'latest.json').write_text(json.dumps(clean(payload),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('WEB_PAYLOAD=',site/'data'/'latest.json')
if __name__=='__main__': main()
