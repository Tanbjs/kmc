import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run

def get_or_create_experiment_id(exp_name: str) -> str:
    """
    Search experiment by name if not found create new one
    Args:
        exp_name (str): experiment name

    Returns:
        str: experiment id
    """

    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name) 
    else:
        exp_id = exp.experiment_id

    return exp_id

def search_child_runs(parent_id) -> list[Run]:
    """
    Search for child runs of a given parent run within an experiment.

    Args:
        exp_id (str): The experiment ID to search within.
        parent_id (str): The parent run ID to filter child runs.

    Returns:
        list: A list of child runs.
    """
    exp_id = mlflow.get_run(parent_id).info.experiment_id
    children = mlflow.search_runs([exp_id], filter_string=f'tags.`mlflow.parentRunId` = "{parent_id}"', order_by=["start_time DESC"])
    return children

def get_model_id(run_id: str) -> list[dict[str, int]]:
    """
    Get the model ID of the latest model output from a given run.
    Args:
        run_id (str): The run ID to look up.

    Returns:
        list: A list of model IDs sorted by step in descending order.
    """
    exp_df = search_child_runs(parent_id=run_id)
    if not exp_df.empty:
        lasted_run_id = exp_df['run_id'][0]
    else:
        lasted_run_id = run_id

    last_run = mlflow.get_run(lasted_run_id)
    model_list = [{'model_id': model.model_id, 'step':model.step} for model in last_run.outputs.model_outputs]
    return sorted(model_list, key=lambda x: x['step'], reverse=True)

def gen_child_name(exp_id: str, parent_name: str, parent_id: str):

    client = MlflowClient()
    children = client.search_runs([exp_id], filter_string=f'tags.`mlflow.parentRunId` = "{parent_id}"')

    max_idx = 0
    for r in children:
        name = r.info.run_name
        if name and name.startswith(parent_name + "_"):
            try:
                idx = int(name.split("_")[-1])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass

    child_name = f"{parent_name}_{max_idx + 1}"
    
    return child_name

def ensure_parent_child(exp_id: str, parent_name: str):

    exp = mlflow.get_experiment(exp_id)
    mlflow.set_experiment(experiment_name=exp.name)

    client = MlflowClient()
    root_runs = client.search_runs(experiment_ids=[exp_id])
    candidates = [r for r in root_runs if r.info.run_name == parent_name]

    if candidates:
        parent_run = sorted(candidates, key=lambda r: r.info.start_time or 0, reverse=True)[0]
        parent_id = parent_run.info.run_id
        child_name = gen_child_name(exp_id=exp_id, parent_name=parent_name, parent_id=parent_id)

        with mlflow.start_run(run_id=parent_id):
            with mlflow.start_run(run_name=child_name or "child", nested=True) as child:
                child_id = child.info.run_id
    else:
        with mlflow.start_run(run_name=parent_name) as parent:
            parent_id = parent.info.run_id
            child_name = gen_child_name(exp_id=exp_id, parent_name=parent_name, parent_id=parent_id)
            with mlflow.start_run(run_name=child_name or "child", nested=True) as child:
                child_id = child.info.run_id

    return parent_id, child_id