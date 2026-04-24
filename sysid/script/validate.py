import mlflow, logging, argparse, urllib3, os, dotenv
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
dotenv.load_dotenv()

from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper
from utils.validate import DMDcPredictor, EDMDcPredictor, DeepPredictor
from utils.data import AUVDataFetchService, feature_selection

def load_model(run):
    logged_model_df = mlflow.search_logged_models(experiment_ids=[run.info.experiment_id])
    model = logged_model_df[(logged_model_df['source_run_id'] == run.info.run_id)][(logged_model_df['name'] == 'final_model')]
    loaded_model = mlflow.pyfunc.load_model(model_uri=model['artifact_location'].values[0])
    return loaded_model

def main():
    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help='run id to load model from')
    parser.add_argument("--n_test", type=int, default=10, help='n-step prediction')
    args = parser.parse_args()
    run_id = args.run_id
    n_test = args.n_test

    with mlflow.start_run(run_id=run_id) as run:

        logging.info(f"Started MLflow run with run_id: {run_id}")
        artifact_uri = run.info.artifact_uri
        config = mlflow.artifacts.load_dict(artifact_uri=f"{artifact_uri}/config.json")
        loaded_model = load_model(run)
        model = loaded_model.unwrap_python_model()
        logging.info("Model loaded successfully from MLflow.")
        # Load test dataset
        loader = AUVDataFetchService(
            run_id=run_id,
            s3_endpoint=os.getenv("S3_ENDPOINT_URL"),
            s3_access_key=os.getenv("S3_ACCESS_KEY_ID"),
            s3_secret_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        )
        test_set = loader.load_dataset(stage='test')

        # feature selection
        state_col = [col for col in test_set[0][1].columns if any(all(word in col for word in group) for group in config['feature_selection']['state'])]
        input_col = [col for col in test_set[0][1].columns if any(all(word in col for word in group) for group in config['feature_selection']['input'])]
        output_col = [col for col in test_set[0][1].columns if any(all(word in col for word in group) for group in config['feature_selection']['output'])]
        reorded_state_col = output_col + [col for col in state_col if col not in output_col]    
        test_featured_list = feature_selection(test_set, reorded_state_col, input_col, output_col)
        
        # Validate
        if isinstance(model, DeepModelWrapper):
            predictor = DeepPredictor(model=model, n_test=n_test)
        elif isinstance(model, EDMDcWrapper):
            predictor = EDMDcPredictor(model=model, n_test=n_test)
        elif isinstance(model, DMDcWrapper):
            predictor = DMDcPredictor(model=model, n_test=n_test)
        else:
            raise ValueError("Unsupported model type for validation.")

        predictor.predict(test_featured_list)
        predictor.log_plot()
        predictor.log_score()
        predictor.controlability_check()
        
if __name__ == "__main__":
    main()