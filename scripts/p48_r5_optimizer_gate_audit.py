"""Read-only CPU audit: distinguish a small learned gate from an unregistered parameter."""
import json

from p48_r5_common import OUT,P1Policy,config,write


def main():
    import torch
    torch.set_num_threads(4)
    rows=[]
    for kind in ['legacy','fixed']:
        policy=P1Policy(config(),kind)  # CPU only; no forward, backward or optimizer update.
        preset=policy.config.get_optimizer_preset();optimizer=preset.build(policy.get_optim_params())
        latest=json.loads((OUT/'runs'/kind/'latest.json').read_text())
        saved=torch.load(latest['path'],map_location='cpu',weights_only=False)
        policy.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        gate=policy.model.object_conditioning.gate
        groups=[group for group in optimizer.param_groups if any(p is gate for p in group['params'])]
        assert len(groups)==1 and gate.requires_grad
        state=optimizer.state[gate];assert float(state['step'])==10000
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
        middle=torch.load(OUT/'runs'/kind/'checkpoints/005000/resume.pt',map_location='cpu',weights_only=False)
        rows.append(dict(kind=kind,gate_requires_grad=gate.requires_grad,optimizer_group_count=len(groups),
            optimizer_lr=groups[0]['lr'],optimizer_weight_decay=groups[0]['weight_decay'],
            optimizer_step=float(state['step']),exp_avg=float(state['exp_avg']),exp_avg_sq=float(state['exp_avg_sq']),
            gate_at_5000=float(middle['model']['model.object_conditioning.gate']),gate_at_10000=float(gate.detach()),
            new_optimizer_updates=0,model_weights_modified=False))
        del policy,optimizer,saved,middle
    write(OUT/'optimizer_gate_audit.json',dict(status='GATE_REGISTERED_AND_ADAM_STATE_UPDATED',rows=rows,
        interpretation='Rules out missing optimizer registration. Moments are historical clipped-gradient statistics, not a fresh gradient measurement.',
        scope='CPU read-only; no new training, seed, data or policy evaluation'))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
