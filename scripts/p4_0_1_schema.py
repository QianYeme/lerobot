"""Non-destructive named array mapping for the formal6 preflight; no training."""
import numpy as np

POSITIONS = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
             'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
STATE_FIELDS = POSITIONS + ['gripper.load', 'gripper.curr']


def map_fields(values, source_names, target_names):
    names = ['gripper.curr' if name == 'gripper.current' else name for name in source_names]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate fields after current alias canonicalization')
    array = np.asarray(values)
    if array.shape[-1] != len(names):
        raise ValueError('Field count differs from array dimension')
    mapped = array[..., [names.index(name) for name in target_names]]
    if not np.isfinite(mapped).all():
        raise ValueError('Nonfinite input')
    return mapped
