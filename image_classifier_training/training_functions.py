import json
import pickle
import shutil
import tempfile
import torch

from clearml import Dataset, StorageManager, Task
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.metrics import Precision, Recall, top_k_accuracy
from fastai.vision.all import (
    Categorize,
    ClassificationInterpretation,
    Datasets,
    Interpretation,
    IntToFloatTensor,
    Learner,
    Path,
    PILImage,
    RegexLabeller,
    Resize,
    SaveModelCallback,
    ToTensor,
    URLs,
    aug_transforms,
    error_rate,
    get_image_files,
    load_learner,
    plt,
    untar_data,
    using_attr,
    vision_learner,
)
from sklearn.model_selection import StratifiedKFold


def make_or_get_training_dataset(
    project: str, i_dataset: int, num_samples_per_chunk=500
):
    """make a dataset that adds images on top of it's parent dataset.
    Will create the parent dataset(s) if necessary recursively"""

    new_dataset_name = f"pets_data_{num_samples_per_chunk}_{i_dataset}"
    try:
        the_dataset = Dataset.get(
            dataset_project=project, dataset_name=new_dataset_name
        )
    except ValueError:

        if i_dataset == 0:
            parent_datsets = None
        else:
            parent = make_or_get_training_dataset(
                project, i_dataset - 1, num_samples_per_chunk=500
            )
            assert parent is not None
            parent_datsets = [parent.id]

        # make a batch of images
        manager = StorageManager()
        train_items_file = manager.get_local_copy(
            "gs://clearml-evaluation/data/pets/train_items.json",
        )
        with open(train_items_file, "r") as fp:
            train_items = json.load(fp)
        images_path = untar_data(URLs.PETS) / "images"

        # random.seed() # randomise the seed or this tempfolder may exist when running locally
        tmp_images_path = Path(tempfile.mkdtemp())
        print("tmp_images_path", tmp_images_path)
        for item in train_items[
            i_dataset * num_samples_per_chunk : (i_dataset + 1) * num_samples_per_chunk
        ]:
            shutil.copy(str(images_path / item) + ".jpg", str(tmp_images_path))
        print(
            i_dataset * num_samples_per_chunk, (i_dataset + 1) * num_samples_per_chunk
        )

        # create a dataset with the nes images as a child of it's parent
        the_dataset = Dataset.create(
            dataset_name=new_dataset_name,
            dataset_project=project,
            parent_datasets=parent_datsets,
        )
        the_dataset.add_files(tmp_images_path)
        the_dataset.upload()
        the_dataset.finalize()
        print("make_new_dataset completed")
        shutil.rmtree(tmp_images_path)
    return the_dataset


def _get_splits_(dataset, n_splits):
    items = dataset.list_files()
    labeller = RegexLabeller(pat=r"^(.*)_\d+.jpg$")
    labels = [labeller(item) for item in items]
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    k_splits = kf.split(items, labels)
    return k_splits


def _top_1_accuracy(inp, targ, axis=-1):
    return top_k_accuracy(inp, targ, k=1, axis=axis)


def _top_2_accuracy(inp, targ, axis=-1):
    return top_k_accuracy(inp, targ, k=2, axis=axis)


def _top_3_accuracy(inp, targ, axis=-1):
    return top_k_accuracy(inp, targ, k=3, axis=axis)


def _precision_micro():
    return Precision(average="micro")


def _recall_micro():

    return Recall(average="micro")


def _make_image_transforms(image_resize):

    return [PILImage.create, Resize(image_resize), ToTensor(), IntToFloatTensor()]


def _make_dls(
    clearml_dataset, image_resize, splits, batch_size=64,
):

    dataset_path = Path(clearml_dataset.get_local_copy())
    items = get_image_files(dataset_path)

    labeller = using_attr(RegexLabeller(pat=r"^(.*)_\d+.jpg$"), "name")
    tfms = [
        _make_image_transforms(image_resize=image_resize),
        [labeller, Categorize()],
    ]

    dsets = Datasets(items, tfms, splits=splits)
    dls = dsets.dataloaders(
        batch_size=batch_size,  # after_item=[ToTensor(), IntToFloatTensor()],
        batch_tfms=aug_transforms(size=image_resize),
    )

    return dls


def _make_dl_test(eval_dataset, image_resize, batch_size):

    dls = _make_dls(
        eval_dataset,
        image_resize,
        [range(len(eval_dataset.list_files())), range(len(eval_dataset.list_files()))],
        batch_size,
    )
    return dls.test_dl(dls.items, with_labels=True)


def _make_learner(dls, run_model_path, backbone_name="resnet34"):
    # backbone_fn = pretrained_resnet_34 = partial(timm.create_model, pretrained=True)
    return vision_learner(
        dls,
        backbone_name,
        metrics=[
            error_rate,
            _precision_micro(),
            _recall_micro(),
            # roc_auc,
            _top_1_accuracy,
            _top_2_accuracy,
            _top_3_accuracy,
        ],
        path=run_model_path,
        pretrained=True,
    )


def _save_model(learner):

    # learner.export(learner.path/'learner.pkl')
    dummy_inp = torch.stack([a[0] for a in learner.dls.train_ds[:2]]).cuda()
    torch.jit.save(
        torch.jit.trace(learner.model.cuda(), dummy_inp),
        learner.path / "model_jit_cuda.pt",
    )
    torch.jit.save(
        torch.jit.trace(learner.model.cpu(), dummy_inp.cpu()),
        learner.path / "model_jit_cpu.pt",
    )
    learner.export("learner.pkl")


def train_image_classifier(
    clearml_dataset,
    backbone_name,
    image_resize,
    batch_size,
    run_model_uri,
    run_tb_uri,
    local_data_path="/data",
    num_epochs=2,
):
    run_info = locals()
    run_info["clearml_dataset"] = {
        "id": clearml_dataset.id,
        "name": clearml_dataset.name,
        "project": clearml_dataset.project,
        "num_entries": len(clearml_dataset.file_entries),
    }
    # get splits
    splits = list(_get_splits_(clearml_dataset, 5))[0]

    run_model_path = Path(local_data_path) / run_model_uri
    run_tb_path = Path(local_data_path) / run_tb_uri
    dls = _make_dls(clearml_dataset, image_resize, splits, batch_size)
    run_model_path.mkdir(parents=True, exist_ok=True)
    learner = _make_learner(dls, run_model_path, backbone_name)
    suggestions = learner.lr_find()
    plt.show()
    tb_callback = TensorBoardCallback(
        log_dir=run_tb_path, trace_model=False, log_preds=False
    )
    learner.fine_tune(
        num_epochs,
        suggestions.valley,
        cbs=[SaveModelCallback(every_epoch=False), tb_callback],
    )
    _save_model(learner)  # with_opt=False

    print("sample validation results")
    learner.show_results()

    plt.show()
    run_info["run_model_path"] = run_model_path
    run_info["training_task_id"] = Task.current_task().id

    # force run_info to be json serializeable
    run_info = json.loads(json.dumps(run_info, default=str))
    print("train_image_classifier completed")
    return run_model_path, run_info


def eval_model(
    run_learner_path,
    run_id,
    training_run_info,
    dataset_name,
    dataset_project,
    run_eval_uri,
    image_resize,
    batch_size,
    local_data_path="/data",
):
    print("run_learner_path:", run_learner_path)
    learner = load_learner(Path(run_learner_path / "learner.pkl"), cpu=False)
    learner.model.to(device="cuda")
    learner.eval()

    # TODO provide project and dataset name to function
    eval_dataset = Dataset.get(
        dataset_project=dataset_project, dataset_name=f"pets_evaluation",
    )

    test_dl = _make_dl_test(eval_dataset, image_resize, batch_size)
    # learner.dls = dls
    test_dl = test_dl.to(device="cuda")
    preds, y_true, losses = learner.get_preds(inner=False, dl=test_dl, with_loss=True)

    run_eval_path = Path(local_data_path) / run_eval_uri
    run_eval_path.mkdir(parents=True, exist_ok=True)
    eval_results = {
        "run_id": run_id,
        "training_run_info": training_run_info,
        "run_learner_path": str(learner.path),
        "eval_dataset": {
            "project": dataset_project,
            "name": dataset_name,
            "id": eval_dataset.id,
            "num_entries": len(eval_dataset.file_entries),
        },
        "metrics": {
            "top_1_accuracy": _top_1_accuracy(preds, y_true).tolist(),
            "top_2_accuracy": _top_2_accuracy(preds, y_true).tolist(),
            "top_3_accuracy": _top_3_accuracy(preds, y_true).tolist(),
        },
    }
    print(eval_results)
    # plot top loss cases
    interp = Interpretation(learn=learner, dl=test_dl, losses=losses)
    interp.plot_top_losses(9, figsize=(15, 10))
    plt.show()

    # plot test confusion
    interp = ClassificationInterpretation.from_learner(learn=learner, dl=test_dl)
    interp.plot_confusion_matrix(figsize=(10, 10))
    plt.show()

    # save results to files
    (run_eval_path / "preds").mkdir()
    with open(f"{run_id}.preds.pkl", "wb") as fid:
        pickle.dump({"preds": preds.tolist(), "y_true": y_true.tolist()}, fid)

    (run_eval_path / "evals").mkdir()
    with open(run_eval_path / "evals" / f"{run_id}.eval.json", "w") as fid:
        json.dump(eval_results, fid, default=str, indent=4)

    # add files to dataset
    try:
        model_evals_dataset = Dataset.get(
            dataset_project=dataset_project, dataset_name=f"model_evals",
        )
    except ValueError:
        model_evals_dataset = Dataset.create(
            dataset_project=dataset_project, dataset_name=f"model_evals",
        )
    model_evals_dataset.add_files(run_eval_path)
    model_evals_dataset.upload()

    # return path for upload and results dict for use in ne
    return eval_results, preds
