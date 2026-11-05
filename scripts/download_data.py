"""Download a raw dataset into data/raw/<name>/ using configs/datasets.yaml.
Usage: python scripts/download_data.py yoochoose
Kaggle sources need the kaggle CLI + ~/.kaggle/kaggle.json (Kaggle -> Settings -> Create New Token),
and you may need to accept the dataset's terms on the Kaggle website once.
If anything fails, put the files into the folder by hand: the script only checks they exist."""
import argparse, shutil, subprocess, sys, tarfile, urllib.request, zipfile
from pathlib import Path
import yaml


def locate(folder: Path, fname: str):
    if (folder / fname).exists():
        return True
    hits = list(folder.rglob(fname))
    if hits:
        shutil.move(str(hits[0]), str(folder / fname))
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--config", default="configs/datasets.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))[a.name]
    folder = Path(cfg["dir"]); folder.mkdir(parents=True, exist_ok=True)

    if not all((folder / f).exists() for f in cfg["files"]):
        if cfg["source"] == "kaggle":
            if shutil.which("kaggle") is None:
                sys.exit("kaggle CLI not found: pip install kaggle, then put kaggle.json in ~/.kaggle/")
            subprocess.run(["kaggle", "datasets", "download", "-d", cfg["kaggle_dataset"],
                            "-p", str(folder), "--unzip"], check=True)
        elif cfg["source"] == "url":
            if "PUT_YOUR" in cfg["url"]:
                sys.exit(f"edit the url in {a.config} first")
            target = folder / Path(cfg["url"]).name
            urllib.request.urlretrieve(cfg["url"], target)
            if tarfile.is_tarfile(target):
                tarfile.open(target).extractall(folder)
            elif zipfile.is_zipfile(target):
                zipfile.ZipFile(target).extractall(folder)
        else:
            sys.exit(f"unknown source {cfg['source']}")

    missing = [f for f in cfg["files"] if not locate(folder, f)]
    if missing:
        sys.exit(f"Missing {missing} in {folder}. Download manually and place them there.")
    for f in cfg["files"]:
        print(f"OK  {folder / f}  ({(folder / f).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
