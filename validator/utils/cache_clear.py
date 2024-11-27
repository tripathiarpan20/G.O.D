import os
import shutil


def delete_dataset_from_cache(dataset_name):
    # Convert slashes to triple underscores to match filesystem format (odd no?)
    dataset_name = dataset_name.replace("/", "___")
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    dataset_path = os.path.join(cache_dir, dataset_name)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        print(f"Deleted dataset '{dataset_name}' from cache.")
    else:
        print(f"Dataset '{dataset_name}' not found in cache.")


def delete_model_from_cache(model_name):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    model_path = os.path.join(cache_dir, model_name)

    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print(f"Deleted model '{model_name}' from cache.")
    else:
        print(f"Model '{model_name}' not found in cache.")
