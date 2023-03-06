from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes, Task
from typing import List


@PipelineDecorator.component(
    return_values=["the_dataset"],
    cache=True,
    task_type=TaskTypes.data_processing,
    execution_queue="default",
    packages="./requirements.txt",
)
def dummy_component(
    project, i_dataset: int, num_samples_per_chunk: int = 500
):
    import sys
    #import torch

    sys.path.insert(0, "/src/clearml_evaluation/")
    from image_classifier_training import training_functions
    print("Inside the dummy")

    return "Made it after the dummy"

@PipelineDecorator.component(
    return_values=["the_dataset"],
    cache=True,
    task_type=TaskTypes.data_processing,
    execution_queue="default",
    packages="./requirements.txt",
)
def make_or_get_training_dataset_component(
    project, i_dataset: int, num_samples_per_chunk: int = 500
):
    import sys

    sys.path.insert(0, "/src/clearml_evaluation/")
    from image_classifier_training import training_functions

    return training_functions.make_or_get_training_dataset(
        project, i_dataset, num_samples_per_chunk=num_samples_per_chunk
    )


"""@PipelineDecorator.component(
    return_values=["run_model_path", "run_info"],
    cache=True,
    task_type=TaskTypes.training,
    repo="git@github.com:shpigi/clearml_evaluation.git",
    repo_branch="main",
    packages="./requirements.txt",
)
def train_image_classifier_component(
    clearml_dataset,
    backbone_name,
    image_resize: int,
    batch_size: int,
    run_model_uri,
    run_tb_uri,
    local_data_path,
    num_epochs: int,
):
    import sys

    sys.path.insert(0, "/src/clearml_evaluation/")
    from image_classifier_training import training_functions

    return training_functions.train_image_classifier(
        clearml_dataset,
        backbone_name,
        image_resize,
        batch_size,
        run_model_uri,
        run_tb_uri,
        local_data_path,
        num_epochs,
    )"""


"""@PipelineDecorator.component(
    return_values=["run_eval_path"],
    cache=True,
    task_type=TaskTypes.testing,
    repo="git@github.com:shpigi/clearml_evaluation.git",
    repo_branch="main",
    packages="./requirements.txt",
)
def eval_model_component(
    run_learner_path,
    run_id,
    training_run_info,
    dataset_name,
    dataset_project,
    run_eval_uri,
    image_resize: int,
    batch_size: int,
    local_data_path,
):
    import sys

    sys.path.insert(0, "/src/clearml_evaluation/")
    from image_classifier_training import training_functions

    eval_results, preds = training_functions.eval_model(
        run_learner_path,
        run_id,
        training_run_info,
        dataset_name,
        dataset_project,
        run_eval_uri,
        image_resize,
        batch_size,
        local_data_path,
    )
    Task.current_task().upload_artifact("preds", preds)
    return eval_results"""


"""@PipelineDecorator.component(
    task_type=TaskTypes.custom,
    docker="python:3.9-bullseye",
    execution_queue="default",
    packages=["clearml==1.6.3rc1", "google-cloud-storage>=1.13.2"],
)
def deploy_model_if_better(new_eval_results: dict, kpi_name="top_1_accuracy"):
    from clearml import StorageManager, Task
    import json

    storage_manager = StorageManager()
    deployed_eval_res_path = storage_manager.get_local_copy(
        "gs://clearml-evaluation/lavi-testing/deployed_model/eval_results.json"
    )
    deploy = True
    deployed_model_eval_results = None
    if deployed_eval_res_path is not None:
        with open(deployed_eval_res_path, "r") as fid:
            deployed_model_eval_results = json.load(fid)
        if (
            new_eval_results["metrics"][kpi_name]
            < deployed_model_eval_results["metrics"][kpi_name]
        ):
            deploy = False
    if deploy:
        print("deploying:")
        task = Task.get_task(
            task_id=new_eval_results["training_run_info"]["training_task_id"]
        )
        print("download the new model")
        model_path = storage_manager.get_local_copy(
            task.artifacts["run_model_path"].url
        )

        deployment_url = "gs://clearml-evaluation/lavi-testing/deployed_model/"
        print("saving new model (jit cuda and cpu versions) to deployment location:")
        print(deployment_url)
        for model_file_name in ["model_jit_cpu.pt", "model_jit_cuda.pt"]:
            storage_manager.upload_file(
                f"{model_path}/{model_file_name}",
                f"{deployment_url}/{model_file_name}",
            )

        print("saving new model's evaluation results to deployment location")
        storage_manager.upload_file(
            str(deployed_eval_res_path), f"{deployment_url}/eval_results.json",
        )
    Task.current_task().upload_artifact(
        "deploy_decision",
        {
            "deploy": deploy,
            "previous_eval": deployed_model_eval_results,
            "new_eval": new_eval_results,
        },
    )"""


@PipelineDecorator.pipeline(
    name="fastai_image_classification_pipeline",
    project="lavi-testing",
    target_project="lavi-testing",
    version="0.2",
    multi_instance_support=True,
    add_pipeline_tags=True,  # add pipe: <pipeline_task_id> tag to all component tasks
    abort_on_failure=True,
)
def fastai_image_classification_pipeline(
    run_tags: List[str],
    i_dataset: int,
    backbone_names: List[str],
    image_resizes: List[int],
    batch_sizes: List[int],
    num_train_epochs: int,
):
    pipeline_metadata = locals()
    from clearml import Task
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pipeline_task = Task.current_task()
    run_id = pipeline_task.id
    pipeline_task.add_tags(run_tags)

    print("Made it here")

    project_name = "mohamed-testing"

    training_dataset = dummy_component(
        project=project_name, i_dataset=i_dataset, num_samples_per_chunk=500
    )

    print(f"After the dummy: {training_dataset}")

    """class TaskURIs:
        def __init__(self, project, pipeline_name, run_id):
            path_pref = f"{project}/{pipeline_name}"
            self.tboard = f"{path_pref}/tboard/{run_id}"
            self.models = f"{path_pref}/models/{run_id}"
            self.evaluations = f"{path_pref}/evaluations/{run_id}"

    def _train_and_eval(
        backbone_name,
        image_resize,
        batch_size,
        num_train_epochs,
        training_dataset,
        run_uris,
        sub_run_id,
        project_name,
    ):
        print("train model")
        run_model_path, training_run_info = train_image_classifier_component(
            clearml_dataset=training_dataset,
            backbone_name=backbone_name,
            image_resize=image_resize,
            batch_size=batch_size,
            run_model_uri=run_uris.models,
            run_tb_uri=run_uris.tboard,
            local_data_path="/data",
            num_epochs=num_train_epochs,
        )
        print("evaluate model")
        eval_results = eval_model_component(
            run_learner_path=run_model_path,
            run_id=sub_run_id,
            training_run_info=training_run_info,
            dataset_name="pets_evaluation",
            dataset_project=project_name,
            run_eval_uri=run_uris.evaluations,
            image_resize=image_resize,
            batch_size=int(batch_size * 1.5),
            local_data_path="/data",
        )
        return eval_results

    project_name = "lavi-testing"
    pipeline_name = "fastai_image_classification"

    print("make dataset")
    training_dataset = make_or_get_training_dataset_component(
        project=project_name, i_dataset=i_dataset, num_samples_per_chunk=500
    )

    futures = []
    with ThreadPoolExecutor(max_workers=4,) as executor:
        for i, (backbone_name, image_resize, batch_size) in enumerate(
            zip(backbone_names, image_resizes, batch_sizes)
        ):
            sub_run_id = f"{run_id}_{i}"
            run_uris = TaskURIs(
                project=project_name, pipeline_name=pipeline_name, run_id=sub_run_id
            )
            futures.append(
                executor.submit(
                    _train_and_eval,
                    backbone_name,
                    image_resize,
                    batch_size,
                    num_train_epochs,
                    training_dataset,
                    run_uris,
                    sub_run_id,
                    project_name,
                )
            )
    eval_results = [f.result() for f in as_completed(futures)]
    print("eval_results:", eval_results)

    best_res = None
    for res in eval_results:
        if (
            not best_res
            or res["metrics"]["top_1_accuracy"] > best_res["metrics"]["top_1_accuracy"]
        ):
            best_res = res

    print(f"The best result is {best_res}")
    this_pipeline_task = Task.current_task()
    this_pipeline_task.upload_artifact(
        "best_result", best_res, metadata=pipeline_metadata
    )

    deploy_model_if_better(best_res)
    print("pipeline complete")"""

if __name__ == "__main__":
    #PipelineDecorator.run_locally()
    PipelineDecorator.set_default_execution_queue('default')
    fastai_image_classification_pipeline(
        run_tags=["run_1"],
        i_dataset=0,
        backbone_names=["resnet34"],
        image_resizes=[1],
        batch_sizes=[10],
        num_train_epochs=2,
    )