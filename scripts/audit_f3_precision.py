"""Read-only F3 local fixture precision matrix; never touches robot IO."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle
    source = root / 'source/src/lerobot/policies/act/local_load_regulator.py'
    spec = importlib.util.spec_from_file_location('f3_adapter_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Execute only the original MLP class, not training/data-loading module code.
    tree = ast.parse(args.original.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LocalResidual')
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(args.original), 'exec'), namespace)
    names = ['local/010000.pt', 'validation/local_fixture.npz',
             'source/src/lerobot/policies/act/local_load_regulator.py']
    report = {'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              'sha256': {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names},
              'original_source_sha256': hashlib.sha256(args.original.read_bytes()).hexdigest(),
              'runs': []}
    manifest = json.loads((root / 'manifest.json').read_text())['sha256']
    checked = [name for name in manifest if not name.endswith('.pyc')]
    mismatches = [name for name in checked
                  if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest[name]]
    report['payload_integrity'] = {'checked': len(checked), 'mismatches': mismatches,
                                   'excluded_mutable_pyc': len(manifest)-len(checked)}
    assert not mismatches, mismatches
    with np.load(root / 'validation/local_fixture.npz') as fixture:
        for device, dtype, tf32 in [('cpu', torch.float32, False),
                                    ('cpu', torch.float64, False),
                                    ('cuda', torch.float32, False),
                                    ('cuda', torch.float32, True)]:
            if device == 'cuda' and not torch.cuda.is_available():
                continue
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            model = module.LocalLoadRegulator(root / 'local/010000.pt', device).to(dtype=dtype)
            inputs = [torch.as_tensor(fixture[n], device=device, dtype=dtype)
                      for n in ['raw_load', 'position']]
            nominal = torch.as_tensor(fixture['nominal_native'][:, 5], device=device, dtype=dtype)
            outputs = []
            saved = torch.load(root / 'local/010000.pt', map_location='cpu', weights_only=True)
            original = namespace['LocalResidual'](True, model.bound).to(device=device, dtype=dtype).eval()
            original.load_state_dict(saved['model'], strict=True)
            norm = saved['normalization']
            features = torch.stack([(inputs[0]-norm['load']['mean'])/norm['load']['std'],
                                    (inputs[1]-norm['position']['mean'])/norm['position']['std'],
                                    (nominal-norm['nominal']['mean'])/norm['nominal']['std']], dim=-1)
            with torch.inference_mode():
                for _ in range(3):
                    outputs.append(model(*inputs, nominal).cpu().double().numpy())
                reference = original(features).cpu().double().numpy()
                singles = torch.cat([model(inputs[0][i:i+1], inputs[1][i:i+1], nominal[i:i+1])
                                     for i in range(len(nominal))]).cpu().double().numpy()
                original_singles = torch.cat([original(features[i:i+1])
                                              for i in range(len(nominal))]).cpu().double().numpy()
            original_error = float(np.max(np.abs(reference-outputs[0])))
            single_error = float(np.max(np.abs(original_singles-singles)))
            assert original_error < 1e-4 and single_error < 1e-4
            error = float(np.max(np.abs(outputs[0] - fixture['expected_delta'])))
            report['runs'].append({'device': device, 'dtype': str(dtype), 'tf32': tf32,
                                   'max_reference_error': error, 'passes_original_1e_4': error < 1e-4,
                                   'repeat_max_error': float(np.max(np.abs(np.stack(outputs) - outputs[0]))),
                                   'original_implementation_max_error': original_error,
                                   'batch1_original_max_error': single_error,
                                   'batch1_vs_batch16_max_error': float(np.max(np.abs(singles-outputs[0]))),
                                   'batch1_vs_fixture_max_error': float(np.max(np.abs(singles-fixture['expected_delta']))),
                                   'batch1_output': singles.tolist(),
                                   'output': outputs[0].tolist()})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
