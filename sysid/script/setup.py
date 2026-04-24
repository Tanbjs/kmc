import os, urllib3, argparse, yaml, dotenv, sys, contextlib
dotenv.load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import mlflow
from kmc.utils.mlflow import ensure_parent_child, get_or_create_experiment_id

def main():
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True, help='path to model config file')
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Open config
    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    with contextlib.redirect_stdout(sys.stderr):
        exp_id = get_or_create_experiment_id(exp_name=config['experiment_name'])
        parent_id, child_id = ensure_parent_child(exp_id=exp_id, parent_name=config['method'])

    with mlflow.start_run(run_id=child_id) as run:
        mlflow.log_dict(config, "config.json")

    print(child_id)

if __name__ == "__main__":
    main()