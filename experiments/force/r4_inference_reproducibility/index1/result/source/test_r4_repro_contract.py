"""CPU-only R4 gate/control tests; not CUDA reproducibility evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def functions():
    source=Path(__file__).resolve().parents[2]/'scripts/run_r4_repro.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['compare','configure']],type_ignores=[])
    cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))
    cudnn=SimpleNamespace(benchmark=False,allow_tf32=False,deterministic=False)
    torch=SimpleNamespace(backends=SimpleNamespace(cuda=cuda,cudnn=cudnn),
        set_num_threads=lambda _:None,use_deterministic_algorithms=lambda _:None)
    scope=dict(np=np,LIMIT=.001,torch=torch)
    exec(compile(selected,str(source),'exec'),scope)
    return scope


def test_gate_is_strict_and_checks_future_chunk_not_only_h0():
    compare=functions()['compare']
    a=np.zeros((4,100,6));b=a.copy();b[2,99,5]=.001
    result=compare(a,b)
    assert result['h0_max_native']==0
    assert result['max_native']==.001 and not result['passed']
    b[2,99,5]=.000999
    assert compare(a,b)['passed']


def test_benchmark_off_changes_only_benchmark_relative_to_historical():
    scope=functions();configure=scope['configure'];backends=scope['torch'].backends
    configure('historical')
    assert backends.cudnn.benchmark and backends.cuda.matmul.allow_tf32 and backends.cudnn.allow_tf32
    configure('benchmark_off')
    assert not backends.cudnn.benchmark
    assert backends.cuda.matmul.allow_tf32 and backends.cudnn.allow_tf32
    assert not backends.cudnn.deterministic


def test_nonfinite_output_cannot_pass():
    a=np.zeros((1,100,6));b=a.copy();b[0,0,0]=np.nan
    assert not functions()['compare'](a,b)['passed']
