# Dataset layout (clean)

This folder contains the data used by the matching + clustering pipeline.

## Active inputs (used by configs/default.yaml)
```
dataset/
  timegrid_features/          # 5-minute timegrid with engineered features (parquet dataset)
  outputs_outcomes/
    outcomes_by_stay.csv       # base outcomes (mortality/VM)
    outcomes_by_stay_full.csv  # derived outcomes (RRT, vasopressor, LOS, etc.)
```

## Archived (not used by current pipeline)
```
dataset/_archive/
  raw_parquets/                # older raw parquet shards (pre-engineering)
  legacy_pipeline/             # previous pipeline (kept for reference)
```

If you do not need the archived data, you can delete the `_archive/` folder.
