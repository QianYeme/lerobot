# Leakage audit

Train36 statistics and gate frozen before F2.2 Dev performance. Final12 samples never requested by these runners. Dev used for checkpoint selection and reporting (selection bias disclosed). Episode/frame IDs only index observations; gate uses raw position at t only. Same-frame recorded target is descriptive in plots, not policy/gate input. No GT contact/manual phase labels or progress input.

Original dataset state contains load/current, but runner explicitly reconstructs six positions and separate single normalized load; current and cached W15 are not policy inputs. Full source/split/norm/gate hashes preserved. This is an implementation audit, not a system-call proof that physical parquet pages contain no other rows.
