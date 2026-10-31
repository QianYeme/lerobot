# PROTOTYPE_SEMANTICS_BLOCKED

No prototypes constructed. The pinned public features contain only two RGB videos,8D robot state,7D action,timestamp/frame/episode/global/task indices. No target masks,boxes,patch IDs or train object segmentation annotations exist in the repository. Task language identifies alphabet soup semantically but does not give its image patches. Simulator eval-init images/privileged segmentation are not training ground truth. SO101 cup bank is prohibited and was not read/reused.

Frozen mechanism remains DINOv2-S(main only,frozen),4×384 domain prototypes,1global+4fixed2×2local tokens,cosine matching,tau.20,xy/confidence,adapter,cross-attention ACT encoder Q/object K,V,gated residual alpha0. Do not create arbitrary k-means/center patches and call them original P1. Phase2 explicitly requires STOP and a separately reviewed deterministic unsupervised strategy if training-region annotations are unavailable.

See blocking_proposal.md. Proposed algorithm,split,source frames/patches and resulting bank hashes are not executed/approved. No model parameter counts or G2 forward/backward PASS is claimed.
