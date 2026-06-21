# upload.py
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id="sameerkhaliq/agentic-rag-assistant",
    repo_type="space",
    ignore_patterns=[".git", ".git/*"],
)