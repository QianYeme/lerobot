"""Parse documentation commands without executing them or accessing hardware."""
import importlib.util
import re
import shlex
import subprocess
import sys
from pathlib import Path

guide = Path(sys.argv[1]).read_text(encoding='utf-8')
source = Path(sys.argv[2])
sys.path.insert(0, str(source / 'src'))
spec = importlib.util.spec_from_file_location('control_f3', source / 'src/lerobot/scripts/control_f3.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
blocks = re.findall(r'```bash\n(.*?)```', guide.split('## 12.', 1)[1], re.S)
assert len(blocks) == 4
for model, block in zip(['M0', 'M1', 'M2', 'M3'], blocks):
    subprocess.run(['bash', '-n'], input=block, text=True, check=True)
    command = block.split('python src/lerobot/scripts/control_f3.py', 1)[1].split('2>&1', 1)[0]
    args = control.parser().parse_args(shlex.split(command.replace('\\\n', '')))
    assert args.model == model and args.trial_id == 'R01_' + model and args.execute
    assert args.robot_port == '/dev/ttyACM0' and args.robot_id == 'nn'
    assert args.robot_max_relative_target == 5 and args.policy_n_action_steps == 1
    assert args.stage0_report and (args.shadow_review is not None) == (model == 'M3')
    assert args.ensemble is None and args.dataset_num_episodes == 1
    print(model, 'BASH_SYNTAX_AND_CLI_PARSE_PASS; hardware NOT_CONNECTED')
