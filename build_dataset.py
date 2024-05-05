import os
import random
from PIL import Image

from datasets import (
    Dataset,
    Features,
    ClassLabel,
    Image as DatasetsImage,
    DatasetDict,
    dataset_dict,
)

MAX_PER_SOURCE = 8000

# these dirs are created by downloading the two datasets from Kaggle
FLICKR8K_DIR = "./flickr8k/Images"
RVL_CDIP = "./rvl-cdip"


def get_images():
    count = 0

    # Processing pictures
    for file in os.listdir(FLICKR8K_DIR):
        if file.endswith(".jpg"):
            yield {
                "image": Image.open(os.path.join(FLICKR8K_DIR, file)).convert("RGB"),
                "is_document": "no",
            }
            count += 1
            if count >= MAX_PER_SOURCE:
                break

    # Processing documents
    dir_paths = [
        os.path.join(RVL_CDIP, d)
        for d in os.listdir(RVL_CDIP)
        if os.path.isdir(os.path.join(RVL_CDIP, d))
    ]

    all_files = []
    for dir_path in dir_paths:
        all_files.extend(
            [
                os.path.join(dir_path, file)
                for file in os.listdir(dir_path)
                if file.lower().endswith(".tif")
            ]
        )
    random.shuffle(all_files)

    for file_path in all_files[:MAX_PER_SOURCE]:
        yield {
            "image": Image.open(file_path).convert("RGB"),
            "is_document": "yes",
        }


features = Features(
    {"image": DatasetsImage(), "is_document": ClassLabel(names=["no", "yes"])}
)


def create_dataset():
    dataset = Dataset.from_generator(get_images, features=features)
    dataset = dataset.shuffle(seed=42)
    split_ratios = {"train": 0.8, "test": 0.1, "validation": 0.1}
    splits = dataset.train_test_split(
        test_size=split_ratios["test"] + split_ratios["validation"]
    )
    test_validation = splits["test"].train_test_split(
        test_size=split_ratios["validation"]
        / (split_ratios["test"] + split_ratios["validation"])
    )

    dataset_dict = DatasetDict(
        {
            "train": splits["train"],
            "test": test_validation["train"],
            "validation": test_validation["test"],
        }
    )
    return dataset_dict


if __name__ == "__main__":
    dataset_dict = create_dataset()
    dataset_dict.save_to_disk("./dataset")
    dataset_dict.push_to_hub("mozilla/docornot")
