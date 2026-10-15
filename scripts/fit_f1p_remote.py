"""Fixed remote F1P probe worker; only consumes the frozen twelve-episode feature bundle."""
import json
import os
os.environ.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
import pickle
import platform
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(sys.argv[1])
job = json.loads((BASE / 'job.json').read_text())
OUT = BASE / 'output'
OUT.mkdir(exist_ok=True)
if list(OUT.glob('probe_*.pkl')) or (OUT / 'done.json').exists():
    raise RuntimeError('Existing fitted outputs: no repeated fit/search')


def task(pair):
    width, probe = pair
    columns = job['columns'][probe]
    fit = [np.load(BASE / f'features_ep{ep:03d}_w{width}.npz') for ep in job['fit']]
    x = np.concatenate([z['x'][:, columns] for z in fit])
    y = np.concatenate([z['y'] for z in fit])
    for z in fit:
        z.close()
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1, penalty='l2', class_weight='balanced', solver='lbfgs',
                               max_iter=2000, tol=1e-4, random_state=20261015)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    assert np.isfinite(model.coef_).all() and model.n_iter_.max() < 2000
    with (OUT / f'probe_w{width}_{probe}.pkl').open('wb') as stream:
        pickle.dump({'scaler': scaler, 'model': model, 'columns': columns, 'fit_episodes': job['fit']}, stream)
    for ep in job['fit'] + job['eval']:
        with np.load(BASE / f'features_ep{ep:03d}_w{width}.npz') as z:
            probability = model.predict_proba(scaler.transform(z['x'][:, columns]))[:, 1]
            assert np.isfinite(probability).all()
            np.savez_compressed(OUT / f'pred_ep{ep:03d}_w{width}_{probe}.npz', frames=z['frames'],
                                y=z['y'], pred=(probability >= .5).astype(int), probability=probability)
    print(f'completed W={width} {probe}', flush=True)
    return {'window': width, 'probe': probe, 'n_fit_windows': len(y),
            'iterations': model.n_iter_.tolist(), 'scaler_n_samples_seen': int(scaler.n_samples_seen_),
            'scaler_mean': scaler.mean_.tolist(), 'converged': True}


import hashlib
for name, expected in job['feature_sha256'].items():
    assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected
pairs = [(w, p) for w in [1, 9, 15] for p in job['columns']]
assert len(pairs) == 12
with threadpoolctl.threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=4) as pool:
    health = list(pool.map(task, pairs))
(OUT / 'health.json').write_text(json.dumps(health, indent=2))
(OUT / 'done.json').write_text(json.dumps({'fits': 12, 'python': platform.python_version(), 'numpy': np.__version__,
    'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'threadpoolctl': threadpoolctl.__version__,
    'fit_episodes': job['fit'], 'eval_episodes': job['eval'], 'policy_training': False}, indent=2))
