import argparse

import palimpzest as pz

# define columns for datasets
text_animal_cols = [
    {"name": "animal", "type": str, "desc": "The type of animal mentioned in the text"},
    {"name": "color", "type": str, "desc": "The color of the animal mentioned in the text"},
]
image_animal_cols = [
    {"name": "animal", "type": str, "desc": "The type of animal in the image"},
    {"name": "color", "type": str, "desc": "The color of the animal in the image"},
]

# query plans
def run_text_join():
    """Build a plan that joins two datasets"""
    ds1 = pz.TextFileDataset(id="animals1", path="join-data/animal-texts/").sem_map(text_animal_cols)
    ds2 = pz.TextFileDataset(id="animals2", path="join-data/animal-texts/").sem_map(text_animal_cols)
    ds3 = ds1.sem_join(ds2, condition="both animals are canines with the same color")
    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
    )
    data_record_collection = ds3.run(config)
    print(data_record_collection.to_df())


def run_image_join():
    """Build a plan that joins two datasets with images"""
    ds1 = pz.ImageFileDataset(id="animals1", path="join-data/animal-images/").sem_map(image_animal_cols)
    ds2 = pz.ImageFileDataset(id="animals2", path="join-data/animal-images/").sem_map(image_animal_cols)
    ds3 = ds1.sem_join(ds2, condition="both animals are canines with the same color")
    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
    )
    data_record_collection = ds3.run(config)
    print(data_record_collection.to_df())


def run_text_image_join():
    """Build a plan that joins a dataset with text to a dataset with images"""
    ds1 = pz.TextFileDataset(id="animals1", path="join-data/animal-texts/").sem_map(text_animal_cols)
    ds2 = pz.ImageFileDataset(id="animals2", path="join-data/animal-images/").sem_map(image_animal_cols)
    ds3 = ds1.sem_join(ds2, condition="both animals are canines with the same color")
    config = pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
    )
    data_record_collection = ds3.run(config)
    print(data_record_collection.to_df())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Palimpzest join demo.")
    parser.add_argument("--task", type=str, help="Which join demo to run")
    args = parser.parse_args()

    if args.task == "text-join":
        run_text_join()
    elif args.task == "image-join":
        run_image_join()
    elif args.task == "text-image-join":
        run_text_image_join()
    else:
        print("Please provide a valid task: one of 'text-join', 'image-join', 'text-image-join'")
        exit(1)
