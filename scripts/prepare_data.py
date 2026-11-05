"""Raw file -> data/processed/<name>/{train,val,test}.pkl + meta.json
Usage: python scripts/prepare_data.py configs/yoochoose_f64.yaml"""
import sys, json
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.loaders import load_raw
from sessrec.data.prepare import build_splits, save

cfg = yaml.safe_load(open(sys.argv[1]))
df = load_raw(cfg["loader"], cfg["raw_path"])
out, stats = build_splits(
    df, recent_fraction=cfg["recent_fraction"], val_frac=cfg["val_frac"], test_frac=cfg["test_frac"],
    min_item_support=cfg["min_item_support"], min_session_len=cfg["min_session_len"],
    max_session_len=cfg["max_session_len"])
save(out, stats, f"data/processed/{cfg['name']}")
print(json.dumps(stats, indent=2))
