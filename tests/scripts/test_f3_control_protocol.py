import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('f3_protocol',Path(__file__).parents[2]/'src/lerobot/utils/f3_control_protocol.py')
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)


def test_fingerprint_ignores_dict_order_but_not_deployment_changes():
    assert protocol.fingerprint(dict(fps=30,clip=5))==protocol.fingerprint(dict(clip=5,fps=30))
    assert protocol.fingerprint(dict(fps=30))!=protocol.fingerprint(dict(fps=20))


@pytest.mark.parametrize('model,mode,trial',[('M0','disabled','R01_M0'),('M3','shadow','SHADOW_03'),('M3','regulated','R03_M3')])
def test_authorized_trial_ids(model,mode,trial):
    protocol.check_trial(model,mode,trial)


@pytest.mark.parametrize('model,mode,trial',[('M3','disabled','R01_M3'),('M0','regulated','R01_M0'),('M3','shadow','SHADOW_04'),('M2','disabled','R01_M1')])
def test_reject_unplanned_trial(model,mode,trial):
    with pytest.raises(ValueError):protocol.check_trial(model,mode,trial)


def test_unreviewed_and_mismatched_gate_rejected():
    with pytest.raises(ValueError):protocol.check_review(dict(passed=False,reviewer='owner',deployment_fingerprint='x'),'x')
    with pytest.raises(ValueError):protocol.check_review(dict(passed=True,reviewer='owner',deployment_fingerprint='other'),'x')


def test_shadow_requires_three_reviewed_trials():
    review=dict(passed=True,reviewer='owner',deployment_fingerprint='x',valid_trial_ids=['SHADOW_01'])
    with pytest.raises(ValueError):protocol.check_review(review,'x',shadow=True)
    review['valid_trial_ids']=['SHADOW_01','SHADOW_02','SHADOW_03']
    protocol.check_review(review,'x',shadow=True)
