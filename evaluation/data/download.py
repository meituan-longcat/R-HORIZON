from huggingface_hub import snapshot_download
from pathlib import Path

datasets = [
    "meituan-longcat/R-HORIZON-Math500",
    "meituan-longcat/R-HORIZON-AIME24",
    "meituan-longcat/R-HORIZON-AIME25",
    "meituan-longcat/R-HORIZON-AMC23",
    "meituan-longcat/R-HORIZON-Websearch",
]

for repo_id in datasets:
    dataset_name = repo_id.split("/")[-1]
    local_dir = Path("./evaluation/data") / dataset_name
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir)
    )