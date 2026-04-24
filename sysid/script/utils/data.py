from typing import List, Tuple, Optional, Dict
import mlflow, logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class AUVDataFetchService:
    """
    Handles fetching datasets from MLflow/S3 and preprocessing robotics data (Quaternion to Euler).
    """
    
    def __init__(self, run_id: str, s3_endpoint: str, s3_access_key: str, s3_secret_key: str):
        self.run_id = run_id
        self.s3_opts = {
            "client_kwargs": {"endpoint_url": s3_endpoint, "verify": False},
            "key": s3_access_key,
            "secret": s3_secret_key
        }
        self.quat_order = ['x', 'y', 'z', 'w']

    def load_dataset(self, stage: str = 'train') -> List[Tuple[str, pd.DataFrame]]:
        """
        Fetches dataset artifacts from MLflow run based on the stage (train/val/test).
        """
        valid_stages = {'train', 'val', 'test'}
        if stage not in valid_stages:
            raise ValueError(f"Stage must be one of {valid_stages}")

        logging.info(f"Loading '{stage}' datasets from Run ID: {self.run_id}...")
        
        run = mlflow.get_run(self.run_id)
        input_list = run.inputs.dataset_inputs
        loaded_datasets = []

        for ds_input in input_list:
            # Check context tag (train/val/test)
            context_tag = next((tag.value for tag in ds_input.tags if tag.key == 'mlflow.data.context'), None)
            
            if context_tag == stage:
                try:
                    dataset_name, df = self._fetch_csv_artifact(ds_input.dataset)
                    df_processed = self._inject_euler_angles(df)
                    loaded_datasets.append((dataset_name, df_processed))
                    logging.info(f"Loaded and processed: {dataset_name}")
                except Exception as e:
                    logging.error(f"Failed to load dataset: {e}")

        return loaded_datasets

    def _fetch_csv_artifact(self, dataset_info) -> Tuple[str, pd.DataFrame]:
        """Internal helper to read CSV from S3 via MLflow source."""
        source = mlflow.data.get_source(dataset_info)
        df = pd.read_csv(source.uri, storage_options=self.s3_opts)
        return dataset_info.name, df

    def _inject_euler_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts Quaternion columns to Euler angles (Roll, Pitch, Yaw) for both Reference and Odometry.
        Modifies DataFrame in-place.
        """
        targets = [
            {
                'prefix': 'ref', 
                'col_pattern': 'orientation', 
                'out_prefix': 'ref_filtered.orientation.euler'
            },
            {
                'prefix': 'odom', 
                'col_pattern': 'orientation', 
                'out_prefix': 'odom_filtered.pose.pose.orientation.euler'
            }
        ]

        for target in targets:
            try:
                # Extract quaternion matrix [N, 4]
                quats = self._extract_ordered_quat(df, target['prefix'], target['col_pattern'])
                
                # Convert to Euler
                r = R.from_quat(quats)
                euler_angles = r.as_euler('xyz', degrees=False) # shape [N, 3]

                # Assign back to DataFrame
                df[f"{target['out_prefix']}.roll"] = euler_angles[:, 0]
                df[f"{target['out_prefix']}.pitch"] = euler_angles[:, 1]
                df[f"{target['out_prefix']}.yaw"] = euler_angles[:, 2]
                
            except ValueError as e:
                logging.warning(f"Skipping Euler conversion for {target['prefix']}: {e}")

        return df

    def _extract_ordered_quat(self, df: pd.DataFrame, prefix: str, pattern: str) -> np.ndarray:
        """Extracts quaternion columns in strictly [x, y, z, w] order."""
        # Find columns matching prefix and pattern
        candidates = df.filter(like=prefix).filter(like=pattern).columns
        
        sorted_cols = []
        for q_comp in self.quat_order:
            # Find column ending with .x, .y, .z, or .w
            col = next((c for c in candidates if c.endswith(f".{q_comp}")), None)
            if col is None:
                raise ValueError(f"Missing quaternion component '{q_comp}' for prefix '{prefix}'")
            sorted_cols.append(col)
            
        return df[sorted_cols].to_numpy()

def prepare_data(dataset_list):
    data_prepared = []
    for file_key, state_df, input_df, output_df in dataset_list:
        Xk = state_df.iloc[:-1].to_numpy()
        Uk = input_df.iloc[:-1].to_numpy()
        X_next = output_df.iloc[1:].to_numpy()
        data_prepared.append((file_key, Xk, Uk, X_next))
    # concat to numpy matrices
    Xk_all = np.concatenate([item[1] for item in data_prepared], axis=0)
    Uk_all = np.concatenate([item[2] for item in data_prepared], axis=0)
    X_next_all = np.concatenate([item[3] for item in data_prepared], axis=0)
    return Xk_all, Uk_all, X_next_all

class AUVLazyDataset(Dataset):
    def __init__(self, x_list, u_list, seq_len, t_list=None):
        self.seq_len = seq_len
        self.samples = []
        
        for traj_idx, (x, u) in enumerate(zip(x_list, u_list)):
            num_ticks = len(x)
            max_start = num_ticks - (seq_len + 1)
            for t_idx in range(max_start + 1):
                self.samples.append((traj_idx, t_idx))

        self.x_data = [torch.tensor(x, dtype=torch.float32) for x in x_list]
        self.u_data = [torch.tensor(u, dtype=torch.float32) for u in u_list]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_id, start_t = self.samples[idx]
        end_t = start_t + self.seq_len 
        x_chunk = self.x_data[traj_id][start_t : end_t]
        u_chunk = self.u_data[traj_id][start_t : end_t]
        x_next_chunk = self.x_data[traj_id][start_t + 1 : end_t + 1]

        return x_chunk, u_chunk, x_next_chunk
    
def feature_selection(data_list, state_col, input_col, output_col):

    featured_list = []
    for file_key, df in data_list:
        state_df = df[state_col]
        input_df = df[input_col]
        output_df = df[output_col]
        featured_list.append((file_key, state_df, input_df, output_df))

    return featured_list