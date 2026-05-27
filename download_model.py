from huggingface_hub import snapshot_download

snapshot_download("medalpaca/medalpaca-7b", local_dir="/model")