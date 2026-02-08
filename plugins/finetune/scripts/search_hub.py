import argparse
import sys
from huggingface_hub import HfApi, DatasetCard, ModelCard
from huggingface_hub.utils import RepositoryNotFoundError

def validate_dataset(dataset_id):
    api = HfApi()
    try:
        info = api.dataset_info(dataset_id)
        print(f"✅ Dataset found: {dataset_id}")
        print(f"--- Details ---")
        print(f"Downloads: {info.downloads}")
        print(f"Likes: {info.likes}")
        # Try to get subsets/configs
        try:
            from datasets import get_dataset_config_names
            configs = get_dataset_config_names(dataset_id)
            print(f"Available configs: {configs}")
        except:
            pass
        return True
    except Exception as e:
        print(f"❌ Dataset not found or inaccessible: {dataset_id}")
        return False

def validate_model(model_id):
    api = HfApi()
    try:
        info = api.model_info(model_id)
        print(f"✅ Model found: {model_id}")
        print(f"--- Details ---")
        print(f"Downloads: {info.downloads}")
        print(f"Likes: {info.likes}")
        print(f"Pipeline: {info.pipeline_tag}")
        return True
    except Exception as e:
        print(f"❌ Model not found or inaccessible: {model_id}")
        return False

def search_datasets(query, limit=5):
    api = HfApi()
    print(f"--- Searching Datasets for: '{query}' ---")
    datasets = api.list_datasets(search=query, limit=limit)
    found = False
    for ds in datasets:
        print(f"- {ds.id} (Downloads: {ds.downloads}, Likes: {ds.likes})")
        found = True
    if not found:
        print("No datasets found matching the query.")

def search_models(query, limit=5):
    api = HfApi()
    print(f"--- Searching Models for: '{query}' ---")
    models = api.list_models(search=query, limit=limit)
    found = False
    for model in models:
        print(f"- {model.id} (Downloads: {model.downloads}, Likes: {model.likes})")
        found = True
    if not found:
        print("No models found matching the query.")

def main():
    parser = argparse.ArgumentParser(description="Search and validate Hugging Face assets")
    parser.add_argument("--dataset", type=str, help="Dataset ID or search query")
    parser.add_argument("--model", type=str, help="Model ID or search query")
    parser.add_argument("--search", action="store_true", help="Perform a search instead of direct validation")

    args = parser.parse_args()

    if args.dataset:
        if args.search:
            search_datasets(args.dataset)
        else:
            if not validate_dataset(args.dataset):
                search_datasets(args.dataset)

    if args.model:
        if args.search:
            search_models(args.model)
        else:
            if not validate_model(args.model):
                search_models(args.model)

if __name__ == "__main__":
    main()
