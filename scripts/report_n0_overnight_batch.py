#!/usr/bin/env python
"""Produce the frozen N0 batch ranking without selecting a winner automatically."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--result-root',required=True); a=ap.parse_args(); root=Path(a.result_root)
    labels=['R2_ACTION_LEVEL','R1_RELATIONAL','R3_FILM']; records={}
    for label in labels:
        p=root/label/'100000'/'metrics.json'
        if not p.exists(): continue
        d=json.loads(p.read_text()); m=d['metrics']; cf={k:abs(m[k]['all']-m['C0_normal']['all']) for k in ('C1_spatial_zero','C2_spatial_shuffle','C3_x_transplant','C4_y_transplant','C5_full_spatial_transplant')}; records[label]={'checkpoint':d['checkpoint'],'dev_l1':m['C0_normal']['all'],'arm_l1':m['C0_normal']['arm'],'gripper_l1':m['C0_normal']['gripper'],'spatial_usage_max':max(cf.values()),'counterfactual':cf}
    ranking=[]
    for label,d in records.items():
        tp = root/label/'transplant'/'metrics.json'
        horizontal = longitudinal = None
        status = 'awaiting_token_transplant'
        if tp.exists():
            tm = json.loads(tp.read_text())
            # The frozen mechanism gate uses the whole-token variant and the
            # direction rates on the two pre-registered spatial axes.
            whole = tm.get('whole', {})
            horizontal = whole.get('horizontal', {}).get('horizontal_direction', {}).get('rate')
            longitudinal = whole.get('longitudinal', {}).get('longitudinal_direction', {}).get('rate')
            status = 'transplant_evaluated'
        ranking.append({'model':label,'horizontal':horizontal,'longitudinal':longitudinal,'spatial_usage':d['spatial_usage_max'],'dev_l1':d['dev_l1'],'status':status})
    (root/'ranking.json').write_text(json.dumps({'ordering_priority':['horizontal mechanism','longitudinal preservation','overall Dev action quality','mechanism cleanliness','resource cost'],'candidates':ranking},indent=2))
    (root/'batch_validation.json').write_text(json.dumps({'experiment':'N0_OVERNIGHT_SPATIAL_CONDITIONING_BATCH','candidates_frozen':labels,'models_evaluated':list(records),'final_test_used':False,'real_robot_used':False,'status':'complete' if records and all(r['status']=='transplant_evaluated' for r in ranking) else 'pending_transplant'},indent=2))
    text='# N0 Overnight Spatial Conditioning Batch\n\nDev-only Formal4 batch. Final Test and real robot were not used.\n\nModel | Dev L1 | Horizontal | Longitudinal | Spatial Usage | Status\n---|---:|---:|---:|---:|---\n'
    for row in ranking:
        h = 'pending' if row['horizontal'] is None else f"{row['horizontal']:.3f}"
        l = 'pending' if row['longitudinal'] is None else f"{row['longitudinal']:.3f}"
        text+=f"{row['model']} | {row['dev_l1']:.6f} | {h} | {l} | {row['spatial_usage']:.6g} | {row['status']}\n"
    text+='\nToken-transplant direction rates are diagnostic only; no automatic winner selection is made. Final Test and real-robot evidence are intentionally absent.\n'; (root/'batch_summary.md').write_text(text)
    (root/'proposal.md').write_text('# N0 follow-up proposal\n\nAfter all candidates and transplant diagnostics are complete, user + ChatGPT decide whether to freeze a candidate or stop absolute spatial conditioning.\n')


if __name__=='__main__': main()
