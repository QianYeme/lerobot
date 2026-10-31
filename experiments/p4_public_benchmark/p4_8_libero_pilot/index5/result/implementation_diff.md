# r5 implementation identity and controlled differences

## Native ACT and P1 trunk

The native ACTPolicy/ACTConfig come from unchanged official v0.4.4 source commit8fff0fde7c79f23a93d845d1a50e985de01f8b8a under inherited index1/workspace/lerobot/src. Its CVAE, shared dual-camera ResNet18, encoder/decoder, unweighted7D action L1, KL default, chunk100 and action_steps100 are retained. No SO101 policy checkpoint is loaded.

The P1 extension attaches the original ObjectConditioning module at native ACT encoder's forward-pre-hook: Q is the ACT **encoder input** token sequence, not the encoder output or decoder queries. K/V are the five adapted object tokens; positional-embedding arguments continue unchanged into the native encoder. The original source fusion is also immediately before self.encoder(...), so this preserves the fusion location without editing upstream ACT files.

Object module is an isolated frozen source copy. Top-only DINOv2-S/14 parameters are frozen and stay eval; four shared prototypes; cosine max/tau.20; global+fixed2x2 local pooling; raw387→256→512 adapter; slot embeddings; MultiheadAttention and unbounded signed scalar gate initially0. No FiLM, explicit coordinate token, Force/Load/Current, NewGL or Local regulator.

P1-Legacy sets its DINO input to the already ImageNet-normalized main CNN image; inner DINO normalizes again. P1-Fixed sets it to the original dataset RGB0..1; inner DINO normalizes once. Prototype construction uses single normalization for both. Their full initialization hashes match, including object module and bank. All three common trunk initial hashes and first real-batch CNN input/feature hashes match. Actual DINO effective-input assertions and100-step optimizer/RNG replay succeeded for both P1 variants.

## Recipe and data adaptation

This is an interface-adapted native ACT architecture baseline. It is not a claim that every part of a CLI training pipeline is unchanged: seed1,10k budget, exact shared cached sampler, deterministic float32/TF32-off numerical mode and Train35 statistics are explicitly frozen overrides. AdamW/LR/clip/KL/dropout use ACTConfig defaults for **all three** policies, not a hidden replay of the SO101 gripper-weight3 recipe. No hardcoded gripper index5 or extra gripper weighting is used; LIBERO's gripper is action index6. Effective config/YAML files disclose all actual values.

Train sampler is an independently seeded PCG64 epoch permutation; final partial batch is kept. Checkpoints store epoch/next-batch cursor, Python/NumPy/Torch/CUDA RNG, optimizer state, model, split/bank/stats SHA. Throughput probes are fresh disposable models and never become formal starts. Across serial/two/three workers the same100 updates yielded identical model hashes; two repeats justified triple scheduling, not free-memory guesses.

## Camera correction and offline execution

Dataset videos already follow the upstream180-degree conversion convention. Native training/prototype RGB is used directly; raw simulator observations go through official LiberoProcessor once. r3's extra training flip only affected discarded engineering smoke and is not propagated to any r5 policy. G0/G1/r2/r3/P4.7 files are unchanged.

Prototype construction and coordinates were frozen before Dev9. Dev9 cache and simulator/assets/backbones reside server-side; evaluation enables HF_HUB_OFFLINE/HF_DATASETS_OFFLINE/TRANSFORMERS_OFFLINE. No online service or browser GPT is needed for the benchmark stages. Full10-task/multi-seed study is not authorized by this single-task pilot.
