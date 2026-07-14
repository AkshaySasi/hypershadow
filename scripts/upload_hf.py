"""Upload the HyperShadow dataset and models to the Hugging Face Hub.

Usage (PowerShell):
    $env:HF_TOKEN = "hf_..."           # a WRITE token from hf.co/settings/tokens
    .venv\Scripts\pip install huggingface_hub
    .venv\Scripts\python scripts\upload_hf.py --user YOUR_HF_USERNAME

Creates two repos:
    YOUR_HF_USERNAME/hypershadow          (dataset)
    YOUR_HF_USERNAME/hypershadow-models   (model checkpoints)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set the HF_TOKEN environment variable first")

    api = HfApi(token=token)
    ds_repo = f"{args.user}/hypershadow"
    model_repo = f"{args.user}/hypershadow-models"

    api.create_repo(ds_repo, repo_type="dataset", exist_ok=True)
    for f in ["data/static.npz", "data/static_meta.json",
              "data/temporal.npz", "data/temporal_meta.json"]:
        print("uploading", f)
        api.upload_file(path_or_fileobj=f, path_in_repo=Path(f).name,
                        repo_id=ds_repo, repo_type="dataset")
    api.upload_file(path_or_fileobj="DATASET_CARD.md", path_in_repo="README.md",
                    repo_id=ds_repo, repo_type="dataset")
    api.upload_folder(folder_path="figures", path_in_repo="figures",
                      repo_id=ds_repo, repo_type="dataset")

    api.create_repo(model_repo, exist_ok=True)
    for f in Path("results").glob("*.pt"):
        print("uploading", f)
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=model_repo)
    for f in Path("results").glob("*.json"):
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f"results/{f.name}",
                        repo_id=model_repo)

    print("\ndone:")
    print(f"  https://huggingface.co/datasets/{ds_repo}")
    print(f"  https://huggingface.co/{model_repo}")


if __name__ == "__main__":
    main()
