"""Package only already-frozen F1P features; launch one isolated remote CPU probe job."""
import hashlib
import json
import os
import tarfile
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/force/f1p_contact_signal_pilot/index1/result'
freeze = json.loads((OUT / 'label_freeze.json').read_text())
subset = json.loads((OUT / 'subset_manifest.json').read_text())
failed = Path(freeze['raw'])
assert not list(failed.glob('probe_*.pkl')) and not list(failed.glob('pred_*.npz'))
files = [failed / f'features_ep{ep:03d}_w{w}.npz' for ep in subset['pilot_fit'] + subset['pilot_eval'] for w in [1, 9, 15]]
job = {'fit': subset['pilot_fit'], 'eval': subset['pilot_eval'],
       'columns': json.loads((OUT / 'probe_config.json').read_text())['probe_columns'],
       'label_freeze': freeze, 'feature_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(failed / 'job.json').write_text(json.dumps(job, indent=2))
archive = failed / 'remote_input.tar.gz'
with tarfile.open(archive, 'w:gz') as bundle:
    for p in files + [failed / 'job.json', ROOT / 'scripts/fit_f1p_remote.py', ROOT / 'scripts/run_f1p_remote.sh']:
        bundle.add(p, arcname=p.name)
remote = '/root/autodl-tmp/F1P_CONTACT_SIGNAL_PILOT_' + failed.name
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(os.environ['D2_HOST'], port=int(os.environ['D2_PORT']), username='root', password=os.environ['D2_PASSWORD'], timeout=30)
try:
    _, stdout, stderr = client.exec_command(f'test ! -e {remote} && mkdir -p {remote}')
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    with client.open_sftp() as sftp:
        sftp.put(str(archive), remote + '/remote_input.tar.gz')
    command = f'tar -xzf {remote}/remote_input.tar.gz -C {remote} && screen -dmS f1p_probe_20261015 bash -c "bash {remote}/run_f1p_remote.sh {remote} > {remote}/run.log 2>&1"'
    _, stdout, stderr = client.exec_command(command)
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
finally:
    client.close()
record = {'remote_dir': remote, 'host': os.environ['D2_HOST'], 'port': int(os.environ['D2_PORT']),
          'screen': 'f1p_probe_20261015', 'bundle_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
          'bytes_uploaded': archive.stat().st_size, 'max_fits': 12, 'local_fits_completed_before_remote': 0,
          'reason': 'Local threadpoolctl 2.2.0 crashes before any fit; remote isolated 3.6.0',
          'local_failed_evidence': str(failed), 'credentials_written': False}
(OUT / 'remote_execution.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
