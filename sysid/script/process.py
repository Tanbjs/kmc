import urllib3, mlflow, argparse, logging, re, io
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO)
from minio import Minio
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sklearn.model_selection import train_test_split

def load_experiments_from_minio(client, bucket_name, prefix, cleaner: dict, smoother: dict = None) -> list:
    """
    Load datasets from Minio with optional smoothing filter.
    """
    # --- Safe Dictionary Access for Cleaner ---
    clean_kwargs = cleaner.get('kwargs', {}) if cleaner else {}
    clean_parts = [f"{k}{v}" for k, v in clean_kwargs.items()]
    clean_method = cleaner.get('method', '') if cleaner else ''
    
    # สร้าง Base Regex แยกไว้ก่อน
    clean_regex_base = rf"{clean_method}.*" + ".*".join(clean_parts)
    
    # --- Safe Dictionary Access for Smoother (Optional) ---
    smooth_regex = None
    if smoother and smoother.get('method'):
        smooth_kwargs = smoother.get('kwargs', {})
        smooth_parts = [f"{k}{v}" for k, v in smooth_kwargs.items()]
        smooth_regex = rf"{smoother['method']}.*" + ".*".join(smooth_parts)
        logging.info(f"🔍 Using Smoother Regex: {smooth_regex}")
        
        # ถ้ามี Smoother ก็ให้ Cleaner match แบบปกติ
        clean_regex = clean_regex_base
    else:
        logging.info("ℹ️ No Smoother configuration found. Strict matching Cleaner file only.")
        
        # [จุดที่แก้] ถ้าไม่มี Smoother บังคับว่าต้องจบด้วย .csv หรือ .parquet ทันที ห้ามมี / ต่อท้าย
        clean_regex = clean_regex_base + r"[^\/]*\.(csv|parquet)$"

    logging.info(f"🔍 Using Cleaner Regex: {clean_regex}")

    dataset = []
    objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
    
    for obj in objects:
        key = obj.object_name
        
        # 1. Check Clean Regex
        if not re.search(clean_regex, key):
            continue
            
        # 2. Check Smooth Regex only if smoother is valid
        if smooth_regex and not re.search(smooth_regex, key):
            continue

        # --- Loading Logic ---
        logging.info(f"📥 Loading {key} from Minio...")
        response = None
        try:
            response = client.get_object(bucket_name, key)
            content = response.read()
            
            if key.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(content))
            else:
                df = pd.read_csv(io.BytesIO(content))
            
            dataset.append((key, df))
            logging.info(f"✅ Loaded {key} with shape {df.shape}.")
            
        except Exception as e:
            logging.error(f"❌ Failed to load {key}: {e}")
        finally:
            if response:
                response.close()
                response.release_conn()

    dataset.sort(key=lambda x: x[0])  
    return dataset

def split_and_log_datasets(run_id, bucket_name, exp_list, ratio=[0.7, 0.15, 0.15], random_state=42):
    """
    Splits experiment list into multiple subsets and logs them to MLflow.
    
    Args:
        run_id (str): Existing MLflow run ID to log datasets under.
        bucket_name (str): S3 bucket name where raw data is stored.
        exp_list (list): List of tuples (file_key, dataframe).
        ratios (list): Proportions for splitting (e.g., [0.7, 0.15, 0.15] for Train/Val/Test).
        random_state (int): Seed for reproducibility.
        
    Returns:
        list: List of data subsets corresponding to the provided ratios.
    """
    # Define dataset contexts based on the number of provided ratios
    contexts = ["train", "val", "test"] if len(ratio) == 3 else ["train", "test"]
    
    if sum(ratio) != 1.0:
        logging.warning("Ratios do not sum to 1.0. Normalizing ratios automatically.")
        ratio = [r / sum(ratio) for r in ratio]

    segments = []
    remaining_data = exp_list
    
    # Iterate through ratios to perform sequential splitting
    for i in range(len(ratio) - 1):
        # Calculate relative test_size for the current remaining data pool
        # Example: To get 0.15 from a remaining 0.3, the relative split is 0.5
        current_ratio = ratio[i]
        relative_test_size = 1 - (current_ratio / sum(ratio[i:]))
        
        main_part, split_part = train_test_split(
            remaining_data, 
            test_size=relative_test_size, 
            random_state=random_state
        )
        segments.append(main_part)
        remaining_data = split_part
    
    # Append the final remaining segment (e.g., the Test set)
    segments.append(remaining_data)

    # Log each segment to MLflow tracking server
    for idx, data_list in enumerate(segments):
        context = contexts[idx]
        count = 0
        
        for file_key, df in data_list:
            # Use nested runs to group dataset logs under the parent training run
            with mlflow.start_run(run_id=run_id, nested=True):
                logging.info(f"🚀 Logging {context} source: {file_key}")
                
                # Create MLflow Dataset object for data lineage tracking
                dataset = mlflow.data.from_pandas(
                    df, 
                    source=f's3://{bucket_name}/{file_key}', 
                    name=file_key,
                )
                # Log the dataset with the specific context (train/val/test)
                mlflow.log_input(dataset, context=context) 
            count += 1
        
        logging.info(f"✅ {context.upper()} set registration complete: {count} files.")

    return segments

def main():

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help='parent run id')
    args = parser.parse_args()
    parent_run_id = args.run_id

    with mlflow.start_run(run_id=parent_run_id) as run:
        config = mlflow.artifacts.load_dict(artifact_uri=f"{run.info.artifact_uri}/config.json")

        # create minio client and upload csv files to minio server
        import pandas as pd
        import urllib3

        # Disable InsecureRequestWarning
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Create a custom HTTP client that ignores SSL certificate verification
        http_client = urllib3.PoolManager(
            cert_reqs='CERT_NONE',
            assert_hostname=False
        )

        # Initialize MinIO Client
        client = Minio(
            "s3.amarr.tan",         
            access_key='minio_user',
            secret_key='minio_password',
            secure=True,         
            http_client=http_client
        )
        bucket_name = 'xplorer-mini-data'

        # --- 1. Load Experiments ---
        cleaner = config.get('cleaner', {})
        smoother = config.get('smoother', None)
        
        target_prefix = 'smoothed/' if (smoother and smoother.get('method')) else 'cleaned/'

        exp_list = load_experiments_from_minio(
            client, 
            bucket_name, 
            prefix=target_prefix, 
            cleaner=cleaner, 
            smoother=smoother
        )
        
        split_and_log_datasets(run_id=parent_run_id, 
                               bucket_name=config.get('bucket_name', bucket_name), 
                               exp_list=exp_list, 
                               ratio=config.get('ratio', [0.7, 0.15, 0.15]), 
                               random_state=42)

if __name__ == "__main__":
    main()