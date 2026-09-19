# web/data/series.f32

Real public daily series, not synthetic. Replaces the earlier deterministic
synthetic 640-point placeholder (`tools/make_web_demo_series.py`, no longer
used by `web/worker.js`).

- **Series**: `us_births` — daily US live births, 1969-01-01 to 1988-12-31
  (7305 points, no gaps).
- **Immediate source**: GIFT-Eval (`Salesforce/GiftEval` on the HuggingFace
  Hub), config `us_births/D`, the same benchmark subset already used by this
  repo's GIFT-Eval scoring (`docs/runs/2026-09-19-gifteval-subset.md`). Local
  snapshot: `~/Code/idle-intelligence/models/data/gift-eval/us_births/D/data-00000-of-00001.arrow`
  (not committed here — models/data dir per `CLAUDE.md`'s "weights and
  datasets are never committed").
- **Upstream origin**: the Monash Time Series Forecasting Archive's
  `us_births` dataset (Godahewa et al. 2021, <https://forecastingdata.org/>),
  itself derived from the CDC National Vital Statistics System daily-births
  series as compiled and published by FiveThirtyEight
  (<https://github.com/fivethirtyeight/data/tree/master/births>). GIFT-Eval's
  own snapshot (`dataset_info.json` in the local Arrow cache) does not embed
  a `license`/`citation` field, so treat FiveThirtyEight's published terms
  (CC BY 4.0 for their `data` repo) and the underlying CDC public-domain data
  as the operative license until independently re-verified — flagged here
  rather than asserted with false confidence.
- **Export**: `web/data/series.f32` is little-endian f32, exported directly
  from the Arrow file above via `pyarrow` (see this commit's Python snippet,
  not checked in as a tool since it's a three-line one-off — the durable
  export path is GIFT-Eval's own `Dataset` loader already used by
  `tools/gifteval_subset.py`). `web/data/series_meta.json` records the start
  date (`1969-01-01`), frequency (`D`), and point count for the page's STATUS
  panel.
