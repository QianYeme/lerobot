"""Pure helpers for F3 explicit approval, same-session settings and trial IDs."""
import hashlib
import json
import re


def fingerprint(settings):
    return hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def check_trial(model, mode, trial):
    if model == 'M3' and mode == 'shadow':
        valid = re.fullmatch(r'SHADOW_0[1-3]',trial)
    else:
        valid = re.fullmatch(r'R0[1-3]_'+model,trial)
        if model == 'M3' and mode != 'regulated':
            valid = False
        if model != 'M3' and mode != 'disabled':
            valid = False
    if not valid:
        raise ValueError('Trial ID/mode outside F3-r1 plan')


def check_review(report, expected_fingerprint, shadow=False):
    if report.get('deployment_fingerprint') != expected_fingerprint:
        raise ValueError('Review does not match frozen deployment settings')
    if report.get('passed') is not True or not report.get('reviewer'):
        raise ValueError('Human review has not passed')
    if shadow and set(report.get('valid_trial_ids',[])) != {'SHADOW_01','SHADOW_02','SHADOW_03'}:
        raise ValueError('Three valid reviewed shadow trials required')
