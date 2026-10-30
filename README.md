# Bio-Analysis

Reach vs non-reach decoding from 1 Hz deconvolved two-photon calcium signals (five mice, two FOVs each,
stages D1 D2 D7 D8 M4 M5).

## Current framework (branch `window_any_dedup`)

- `script/lgbm_stages.py`: nested cross-validated LightGBM decoder, fixed window -2..+4 s around the event start,
  non-reaches cleaned by `window_any` (dropped when their window touches a reach) then `dedup` (only the first of
  overlapping non-reach windows kept), 25 per-session reach-selectivity features learned from training labels only.
  `pooled.csv` reports the pooled scores and the `session_balanced` scores (every session weighs the same, half
  reach / half non-reach, so the session reach rate alone scores at chance); `split_by="block"` and
  `circular_shift` give the slow-drift / adjacency check and the shuffled-signal control.
- `script/pre_reach_cells.py`: PRE (-3..-1 s) and POST (0..+4 s) cells per session by the SPS rule, with figures.
- `src_trial/`: the pipeline (`lgbm_nested.py`, `lgbm_cellstats.py`, `cell_select.py`, `deepset_predict.py`
  for `build_spans`, `trial_predict.py` for the event tables).
- `plot/lgbm_nested/cont/fixed2+4_window_any_dedup_*`: the results of the current framework.
