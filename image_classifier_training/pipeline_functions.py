# import torch
# from fastai.vision.all import (
#     partial,
#     Precision,
#     Recall,
#     get_image_files,
#     using_attr,
#     Datasets,
#     vision_learner,
#     L,
#     RegexLabeller,
#     PILImage,
#     Resize,
#     ToTensor,
#     IntToFloatTensor,
#     Categorize,
#     aug_transforms,
#     resnet34,
#     accuracy,
#     error_rate,
#     SaveModelCallback,
#     top_k_accuracy,
#     load_learner,
#     plt,
#     pickle,
#     json,
#     Interpretation,
#     ClassificationInterpretation,
# )

# # # import timm
# from fastai.callback.tensorboard import TensorBoardCallback

# from clearml.automation.controller import PipelineDecorator
# from clearml import TaskTypes


def make_new_dataset(project, i_dataset, num_samples_per_chunk=500):
    import json
    import shutil
    from clearml import StorageManager, Dataset
    from fastai.vision.all import URLs, untar_data

    try:
        the_dataset = Dataset.get(
            dataset_project=project, dataset_name=f"pets_data_{i_dataset}"
        )
    except:
        manager = StorageManager()
        train_items_file = manager.get_local_copy(
            "gs://clearml-evaluation/data/pets/train_items.json",
        )
        with open(train_items_file, "r") as fp:
            train_items = json.load(fp)
        images_path = untar_data(URLs.PETS) / "images"

        new_images_path = images_path / "new_train_images"
        new_images_path.mkdir(exist_ok=True)
        for item in train_items[
            i_dataset * num_samples_per_chunk : (i_dataset + 1) * num_samples_per_chunk
        ]:
            shutil.copy(str(images_path / item) + ".jpg", str(new_images_path))

        print(new_images_path)
        print(
            i_dataset * num_samples_per_chunk, (i_dataset + 1) * num_samples_per_chunk
        )

        if i_dataset == 0:
            parent_datsets = None
        else:
            parent = Dataset.get(
                dataset_project="lavi-testing",
                dataset_name=f"pets_data_{i_dataset - 1}",
            )
            assert parent is not None
            parent_datsets = [parent.id]

        the_dataset = Dataset.create(
            dataset_name=f"pets_data_{i_dataset}",
            dataset_project="lavi-testing",
            parent_datasets=parent_datsets,
        )
        the_dataset.add_files(new_images_path)
        the_dataset.upload()
        the_dataset.finalize()
        print("make_new_dataset completed")
    return the_dataset


def get_splits(dataset, n_splits):
    from fastai.vision.all import RegexLabeller
    from sklearn.model_selection import StratifiedKFold

    items = dataset.list_files()
    labeller = RegexLabeller(pat=r"^(.*)_\d+.jpg$")
    labels = [labeller(item) for item in items]
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    k_splits = kf.split(items, labels)
    return k_splits


def top_1_accuracy(inp, targ, axis=-1):
    from fastai.metrics import top_k_accuracy

    return top_k_accuracy(inp, targ, k=1, axis=axis)


def top_2_accuracy(inp, targ, axis=-1):
    from fastai.metrics import top_k_accuracy

    return top_k_accuracy(inp, targ, k=2, axis=axis)


def top_3_accuracy(inp, targ, axis=-1):
    from fastai.metrics import top_k_accuracy

    return top_k_accuracy(inp, targ, k=3, axis=axis)


def precision_micro():
    from fastai.metrics import Precision

    return Precision(average="micro")


def recall_micro():
    from fastai.metrics import Recall

    return Recall(average="micro")


def make_image_transforms():
    from fastai.vision.all import PILImage, Resize, ToTensor, IntToFloatTensor

    return [PILImage.create, Resize(460), ToTensor(), IntToFloatTensor()]


def make_dls(
    clearml_dataset,
    splits,
):
    from fastai.vision.all import (
        Path,
        Categorize,
        get_image_files,
        using_attr,
        RegexLabeller,
        Datasets,
        aug_transforms,
    )

    dataset_path = Path(clearml_dataset.get_local_copy())
    items = get_image_files(dataset_path)

    labeller = using_attr(RegexLabeller(pat=r"^(.*)_\d+.jpg$"), "name")
    tfms = [
        make_image_transforms(),
        [labeller, Categorize()],
    ]

    dsets = Datasets(items, tfms, splits=splits)
    dls = dsets.dataloaders(  # after_item=[ToTensor(), IntToFloatTensor()],
        batch_tfms=aug_transforms(size=224),
    )

    return dls


def make_dl_test(dataset_project, dataset_name):
    from clearml import Dataset

    eval_dataset = Dataset.get(
        dataset_project="lavi-testing",
        dataset_name=f"pets_evaluation",
    )
    dls = make_dls(
        eval_dataset,
        [range(len(eval_dataset.list_files())), range(len(eval_dataset.list_files()))],
    )
    return dls.test_dl(dls.items, with_labels=True)


def make_learner(dls, run_model_path, backbone_name="resnet34"):
    # backbone_fn = pretrained_resnet_34 = partial(timm.create_model, pretrained=True)
    from fastai.vision.all import vision_learner, accuracy, error_rate

    return vision_learner(
        dls,
        backbone_name,
        metrics=[
            accuracy,
            error_rate,
            precision_micro(),
            recall_micro(),
            # roc_auc,
            top_1_accuracy,
            top_2_accuracy,
            top_3_accuracy,
        ],
        path=run_model_path,
        pretrained=True,
    )


def save_model(learner):
    import torch

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
    clearml_dataset, backbone_name, run_model_uri, run_tb_uri, local_data_path="/data"
):
    from fastai.callback.tensorboard import TensorBoardCallback
    from fastai.vision.all import plt, SaveModelCallback, Path

    # get splits
    splits = list(get_splits(clearml_dataset, 5))[0]

    run_model_path = Path(local_data_path) / run_model_uri
    run_tb_path = Path(local_data_path) / run_tb_uri
    dls = make_dls(clearml_dataset, splits)
    run_model_path.mkdir(parents=True, exist_ok=True)
    learner = make_learner(dls, run_model_path, backbone_name)
    suggestions = learner.lr_find()
    plt.show()
    tb_callback = TensorBoardCallback(
        log_dir=run_tb_path, trace_model=False, log_preds=False
    )
    learner.fine_tune(
        2,
        suggestions.valley,
        cbs=[
            SaveModelCallback(every_epoch=False),
            tb_callback,
        ],
    )
    save_model(learner)  # with_opt=False

    print("sample validation results")
    learner.show_results()
    plt.show()
    print("run_learner_path", learner.path)
    print("train_image_classifier completed")
    return run_model_path, run_tb_path


def eval_model(
    run_learner_path,
    run_id,
    dataset_name,
    dataset_project,
    run_eval_uri,
    local_data_path="/data",
):
    import pickle
    import json
    from fastai.vision.all import (
        Learner,
        load_learner,
        Path,
        Interpretation,
        ClassificationInterpretation,
        plt,
    )

    print("run_learner_path:", run_learner_path)
    learner = load_learner(Path(run_learner_path / "learner.pkl"), cpu=False)
    learner.model.to(device="cuda")
    test_dl = make_dl_test(dataset_project, dataset_name)
    #learner.dls = dls
    test_dl = test_dl.to(device="cuda")
    preds, y_true, losses = learner.get_preds(inner=False, dl=test_dl, with_loss=True)

    run_eval_path = Path(local_data_path) / run_eval_uri
    run_eval_path.mkdir(parents=True, exist_ok=True)
    eval_results = {
        "run_id": run_id,
        "run_learner_path": learner.path,
        "eval_dataset": {
            "dataset_project": dataset_project,
            "dataset_name": dataset_name,
        },
        "metrics": {
            "top_1_accuracy": top_1_accuracy(preds, y_true),
            "top_2_accuracy": top_2_accuracy(preds, y_true),
            "top_3_accuracy": top_3_accuracy(preds, y_true),
        },
    }
    print(eval_results["metrics"])

    interp = Interpretation(learn=learner, dl=test_dl, losses=losses)
    interp.plot_top_losses(9, figsize=(15, 10))
    plt.show()
    interp = ClassificationInterpretation.from_learner(learn=learner, dl=test_dl)
    interp.plot_confusion_matrix(figsize=(10, 10))
    #     clearml_task.logger.report_matplotlib_figure(
    #         title="Manual Reporting - confusion matrix", series=run_id, iteration=0, figure=plt
    #     )

    plt.show()
    with open(run_eval_path / "preds.pkl", "wb") as fid:
        pickle.dump({"preds": preds, "y_true": y_true}, fid)

    with open(run_eval_path / "evaluation_results.json", "w") as fid:
        json.dump(eval_results, fid, default=str, indent=4)
    return run_eval_path
