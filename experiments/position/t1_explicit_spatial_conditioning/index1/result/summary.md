# T1 Explicit Spatial Conditioning Discovery

Dev-only evaluation on the frozen Formal4 split. Final Test and real-robot inference were not used.

## Result: FAIL for this implementation

- B1 Dev first-action L1: `0.132089`.
- P2 Dev first-action L1: `0.145435` (worse than B1).
- Learned beta: `0.0031806775368750095` (finite and non-zero).
- Maximum S1–S4 change in Dev L1: `9.469e-08`; spatial ablations are effectively unchanged.
- Whole-token transplant horizontal consistency: `0.263`; longitudinal consistency: `0.556`. Both are below the frozen `0.70` gate.

The explicit coordinate path was not demonstrated to control the action. This is a valid negative mechanism finding; do not enter Confirmation or real-robot testing.
