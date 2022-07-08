import shutil 
import json
from fastai.vision.all import *
from clearml import StorageManager
import numpy as np


def deterministic_random_split(list_of_str, p1, p2=None):
    """
    Deterministically return a int list of length len(list_of_str) that splits the list.
    The portion of 0's in the returned array should be p1 up to rounding to 3 decimal points.
    The split is performed by pseudo-randomly mapping the string to integers in the interval 0-1000
    and entries whose hash is below p1*1000 are labelled 0.
    p1 is the percentage of data that goes into the training set.
    If p2 is not set, 1-p1 goes to the validation set.
    If p2 is set, p2 goes to the test set, 1-p1-p2 goes to the validation set
    """

    def convert_string_to_value(string, p1, p2):
        hash_value = int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16) % 1000

        threshold1 = p1 * 1000
        threshold2 = 1000
        if p2 is not None:
            threshold2 = (p1 + 1 - p1 - p2) * 1000

        if hash_value < threshold1:
            # If hash value is less than first threshold it belongs to the training set
            return 0
        if hash_value < threshold2:
            # If hash value between threshold 1 and 2 it belongs to the validation set
            return 1
        # If hash value is greater than threshold 2
        return 2

    return [convert_string_to_value(s, p1, p2) for s in list_of_str]

def make_raw_data_and_eval_dataset():
    run_dataset_path = untar_data(URLs.PETS)
    items = get_image_files(run_dataset_path / "images")
    test_items = np.array(items)[
        np.array(deterministic_random_split([item.stem for item in items], p1=0.8)) > 0
    ]
    train_items = np.array(items)[
        np.array(deterministic_random_split([item.stem for item in items], p1=0.8)) == 0
    ]
    assert set.intersection(set(test_items), set(train_items)) == set()
    test_path = run_dataset_path / "test_images"
    test_path.mkdir(exist_ok=True)
    train_path = run_dataset_path / "train_images"
    train_path.mkdir(exist_ok=True)
    for item in test_items:
        shutil.copy(str(item), str(test_path))
    for item in train_items:
        shutil.copy(str(item), str(train_path))

    # upload raw files to gs
    manager = StorageManager()
    manager.upload_folder(
        train_path, "gs://clearml-evaluation/data/pets/raw_incoming_data"
    )
    manager.upload_folder(
        test_path, "gs://clearml-evaluation/data/pets/evaluation_data"
    )

    with open(run_dataset_path/"train_items.json",'w') as fp:
        json.dump([a.stem for a in train_items], fp)
    manager.upload_file(run_dataset_path/"train_items.json","gs://clearml-evaluation/data/pets/train_items.json" )
    print(test_items)
    with open(run_dataset_path/"evaluation_items.json",'w') as fp:
        json.dump([a.stem for a in test_items], fp)
    manager.upload_file(run_dataset_path/"evaluation_items.json","gs://clearml-evaluation/data/pets/evaluation_items.json" )
        
    
    # create an evaluation dataset
    eval_dataset = Dataset.create(
        dataset_name=f"pets_evaulation",
        dataset_project="lavi-testing",
        parent_datasets= None,
    )
    eval_dataset.add_files(test_path)
    eval_dataset.upload()
    eval_dataset.finalize()

