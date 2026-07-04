import argparse
import os

from huggingface_hub import HfApi

# --- CONFIGURATION ---
HF_REPO_ID = "StephanAkkerman/stock-recognizer-model"
# ---------------------


def push_model_to_hub(folder_path, repo_id, version=None, private=True):
    """Uploads a trained adapter folder to the Hugging Face Hub and, if a
    version is given, tags the resulting commit so the engine repo can pin
    a specific `revision`."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Adapter folder not found: {folder_path}")

    api = HfApi()

    print(f"📦 Creating/verifying repo: https://huggingface.co/{repo_id}")
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    commit_message = f"Upload adapter{f' ({version})' if version else ''}"
    print(f"🚀 Uploading {folder_path} to {repo_id}...")
    commit_info = api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    if version:
        print(f"🏷️  Tagging commit as {version}...")
        api.create_tag(
            repo_id,
            tag=version,
            repo_type="model",
            revision=commit_info.oid,
            exist_ok=True,
        )

    print(f"🎉 Model successfully published{' and tagged ' + version if version else ''}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push a trained GLiNER2 adapter folder to the Hugging Face Hub."
    )
    parser.add_argument(
        "folder_path", help="Path to the adapter folder, e.g. models/reddit_adapter_v18/final"
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version tag to apply to the commit, e.g. v18 (used by the engine repo to pin a revision)",
    )
    parser.add_argument("--repo-id", default=HF_REPO_ID, help="Target HF Hub model repo")
    parser.add_argument(
        "--public", action="store_true", help="Make the repo public (default: private)"
    )
    args = parser.parse_args()

    push_model_to_hub(
        args.folder_path, args.repo_id, version=args.version, private=not args.public
    )
