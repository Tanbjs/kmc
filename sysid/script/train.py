import os, mlflow, argparse, logging, urllib3, dotenv
dotenv.load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

import numpy as np
import pandas as pd

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import KFold

from kmc.model import DMDc, EDMDc, LitKAEc
from kmc.utils.observable import PolynomialObservable, CombinedObservable, TrigonometricObservable
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper
from utils.data import AUVDataFetchService, AUVLazyDataset, prepare_data, feature_selection

def normalize_datasets(featured_list, scaler_x, scaler_u, scaler_y):
    # transform train and test sets
    normalized_set = []
    for file_key, state_df, input_df, output_df in featured_list:
        state_normalized = pd.DataFrame(scaler_x.transform(state_df), columns=state_df.columns)
        input_normalized = pd.DataFrame(scaler_u.transform(input_df), columns=input_df.columns)
        output_normalized = pd.DataFrame(scaler_y.transform(output_df), columns=output_df.columns)
        normalized_set.append((file_key, state_normalized, input_normalized, output_normalized))

    return normalized_set

def fit_scaler(train_featured_list, method='standard', args={}):

    # concat all dataframes to fit scaler
    for file_key, state_df, input_df, output_df in train_featured_list:
        # reoder columns first n columns of state is output
        state_concat = state_df if 'state_concat' not in locals() else pd.concat([state_concat, state_df], axis=0)
        input_concat = input_df if 'input_concat' not in locals() else pd.concat([input_concat, input_df], axis=0)
        output_concat = output_df if 'output_concat' not in locals() else pd.concat([output_concat, output_df], axis=0)
    
    # normalize selected columns
    if method == 'standard':
        scaler_x = StandardScaler(**args.get('standard', {}))
        scaler_u = StandardScaler(**args.get('standard', {}))
        scaler_y = StandardScaler(**args.get('standard', {}))
        scaler_x.fit(state_concat)
        scaler_u.fit(input_concat)
        scaler_y.fit(output_concat)
    elif method == 'minmax':
        scaler_x = MinMaxScaler(feature_range=tuple(args.get('feature_range', [-1, 1])))
        scaler_u = MinMaxScaler(feature_range=tuple(args.get('feature_range', [-1, 1])))
        scaler_y = MinMaxScaler(feature_range=tuple(args.get('feature_range', [-1, 1])))
        scaler_x.fit(state_concat)
        scaler_u.fit(input_concat)
        scaler_y.fit(output_concat)

    return scaler_x, scaler_u, scaler_y

def train_dmdc(normalized_train_set, config):

    Xk_train, Uk_train, Xk1_train = prepare_data(normalized_train_set)
    logging.info(f"Xk1_train.shape: {Xk1_train.shape}, Uk_train.shape: {Uk_train.shape}, Xk_train.shape: {Xk_train.shape}")
    logging.info("Training DMDc model...")
    
    # check condition of phi matrix [Xk; Uk]
    phi_matrix = np.concatenate([Xk_train, Uk_train], axis=1)
    cond_number = np.linalg.cond(phi_matrix.T @ phi_matrix)
    logging.info(f"Condition number of phi matrix: {cond_number}")
    mlflow.log_param("phi_condition_number", cond_number)

    if not config['use_ridge']:
        dmdc_model = DMDc(method='ols')        
    else:
       dmdc_model = DMDc(method='ridge', alpha=config['ridge_alpha'])
       phi_ridge = phi_matrix.T @ phi_matrix + config['ridge_alpha'] * np.eye(phi_matrix.shape[1])
       cond_number_ridge = np.linalg.cond(phi_ridge)
       logging.info(f"Condition number after ridge regularization: {cond_number_ridge}")
       mlflow.log_param("phi_ridge_condition_number", cond_number_ridge)

    dmdc_model.fit(Xk_train, Xk1_train, Uk_train)
    return dmdc_model

def train_edmdc(normalized_train_set, config):
    Xk_train, Uk_train, Xk1_train = prepare_data(normalized_train_set)
    logging.info(f"Xk1_train.shape: {Xk1_train.shape}, Uk_train.shape: {Uk_train.shape}, Xk_train.shape: {Xk_train.shape}")
    logging.info("Training EDMDc model...")
    
    observable = config.get('observable', {})
    degree = observable.get('degree', 2)
    if observable.get('type') == 'polynomial':
        poly_obs = PolynomialObservable(degree=degree, include_bias=False)

    options = config.get('options', {})
    reg_method = options.get('method', 'ols')
    reg_args = options.get('regression_args', {})

    Zk_train = poly_obs.fit_transform(Xk_train)
    Omega = np.hstack((Zk_train, Uk_train))
    Gram_matrix = Omega.T @ Omega
    
    cond_before = np.linalg.cond(Gram_matrix)
    logging.info(f"Condition number (Before Regularization): {cond_before:.4e}")
    mlflow.log_param("edmdc_cond_num_before", cond_before)

    if reg_method == 'ridge':
        alpha = reg_args.get('alpha', 1.0)
        Gram_reg = Gram_matrix + alpha * np.eye(Gram_matrix.shape[0])
        cond_after = np.linalg.cond(Gram_reg)
        logging.info(f"Condition number (After Ridge, alpha={alpha}): {cond_after:.4e}")
        mlflow.log_param("edmdc_cond_num_after", cond_after)
    elif reg_method in ['lasso', 'elasticnet']:
        logging.info(f"Condition number (After): N/A (Using L1 penalty: {reg_method})")
        mlflow.log_param("edmdc_cond_num_after", f"N/A ({reg_method})")
    else:
        logging.info("Condition number (After): Same as before (OLS method)")
        mlflow.log_param("edmdc_cond_num_after", cond_before)

    model = EDMDc(obs=poly_obs)
    model.fit(Xk_train, Xk1_train, Uk_train, method=reg_method, **reg_args)
    
    logging.info(f"Using {observable.get('type')} observable with degree {degree} for EDMDc.")
    try:
        logging.info(f"Observable feature names: {model._obs_func.get_output_names()}")
    except AttributeError:
        pass
        
    return model

def train_deep_vanilla(normalized_train_set, normalized_val_set, config, mlflowlogger):

    x_list = [df[1].to_numpy() for df in normalized_train_set]
    u_list = [df[2].to_numpy() for df in normalized_train_set]

    x_val_list = [df[1].to_numpy() for df in normalized_val_set]
    u_val_list = [df[2].to_numpy() for df in normalized_val_set]
    
    seq_len = config.get('sequence_length', 10)
    train_dataset = AUVLazyDataset(x_list, u_list, seq_len) 
    train_dataloader =  DataLoader(train_dataset, batch_size=config.get('batch_size', 64), shuffle=True, drop_last=True)
    val_dataset = AUVLazyDataset(x_val_list, u_val_list, seq_len)
    val_dataloader =  DataLoader(val_dataset, batch_size=config.get('batch_size', 64), shuffle=False, drop_last=True)

    model_config = {
        'feature_dim': x_list[0].shape[1],
        'control_dim': u_list[0].shape[1],
        'latent_dim': config.get('latent_dim', 10),
        'decoder_mode': config.get('decoder_mode', 'linear'),
        'hidden_enc_cfg': config.get('encoder_hidden', {}),
        'hidden_dec_cfg': config.get('decoder_hidden', {}),
    }
    
    model = LitKAEc(
        model_config=model_config,
        optimizer_config=config.get('optimizer1', {}),
        scheduler_config=config.get('scheduler', None),
        loss_weights=config.get('loss_weights', {})
    )

    # Callbacks and Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="valid/total_loss",  
        mode="min",                  
        save_top_k=5,               
        save_last=True,             
        filename="kaec-{epoch:02d}-{valid/total_loss:.4f}",
        auto_insert_metric_name=False
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # early_stop_callback = EarlyStopping(
    #     monitor="valid/total_loss",    
    #     min_delta=0.001,        
    #     patience=10,             
    #     verbose=True,
    #     mode="min"              
    # )

    # Trainer
    trainer = L.Trainer(max_epochs=config.get('max_epochs', 1000), 
                        callbacks=[checkpoint_callback, lr_monitor], 
                        check_val_every_n_epoch=1,
                        gradient_clip_val=1.0,
                        gradient_clip_algorithm="norm",
                        accelerator='auto', 
                        devices='auto', 
                        deterministic=True,
                        logger=mlflowlogger)
    
    trainer.fit(model, train_dataloader, val_dataloader)
    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID")
    args = parser.parse_args()
    
    # Get arguments
    run_id = args.run_id

    # Train model
    with mlflow.start_run(run_id=run_id) as run:

        logging.info(f"Started MLflow run with run_id: {run_id}")
        artifact_uri = run.info.artifact_uri
        config = mlflow.artifacts.load_dict(artifact_uri=f"{artifact_uri}/config.json")
        
        # Load training data
        loader = AUVDataFetchService(
            run_id=run_id,
            s3_endpoint=os.getenv("S3_ENDPOINT_URL"),
            s3_access_key=os.getenv("S3_ACCESS_KEY_ID"),
            s3_secret_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        )
        train_set = loader.load_dataset(stage='train')

        # Feature Extraction from Config
        state_col = [col for col in train_set[0][1].columns if any(all(word in col for word in group) for group in config['feature_selection']['state'])]
        input_col = [col for col in train_set[0][1].columns if any(all(word in col for word in group) for group in config['feature_selection']['input'])]
        output_col = [col for col in train_set[0][1].columns if any(all(word in col for word in group) for group in config['feature_selection']['output'])]
        
        def get_priority(col):
            if 'position' in col: return 0
            if 'orientation' in col: return 1
            if 'twist' in col: return 2
            return 3

        sorted_master = sorted(state_col, key=get_priority)
        reorded_state_col = [c for c in sorted_master if c in output_col] + [c for c in sorted_master if c not in output_col]
        output_col = [c for c in reorded_state_col if c in output_col]

        train_featured_list = feature_selection(train_set, reorded_state_col, input_col, output_col)
        scaler_x, scaler_u, scaler_y = fit_scaler(train_featured_list, method=config.get('scaler_method'), args=config.get('scaler_args', {}))

        mlflow.log_param("scaler_x", scaler_x.__class__.__name__)
        mlflow.log_param("scaler_u", scaler_u.__class__.__name__)
        mlflow.log_param("scaler_y", scaler_y.__class__.__name__)

        # transform train set and val set
        normalized_train_set = normalize_datasets(train_featured_list, scaler_x, scaler_u, scaler_y)
        if len(config['ratio']) == 3:
            val_set = loader.load_dataset(stage='val')
            val_featured_list = feature_selection(val_set, reorded_state_col, input_col, output_col)
            normalized_val_set = normalize_datasets(val_featured_list, scaler_x, scaler_u, scaler_y)

        # Train based on method
        if config['method'] == 'dmdc':
            model = train_dmdc(normalized_train_set, config)
            model_wrapper = DMDcWrapper(model=model, 
                                         state_col=reorded_state_col, 
                                         input_col=input_col,
                                         output_col=output_col, 
                                         scaler_x=scaler_x, 
                                         scaler_u=scaler_u, 
                                         scaler_y=scaler_y)
            
        elif config['method'] == 'edmdc':
            model = train_edmdc(normalized_train_set, config)
            model_wrapper = EDMDcWrapper(model=model, 
                                         state_col=reorded_state_col, 
                                         input_col=input_col,   
                                         output_col=output_col, 
                                         scaler_x=scaler_x, 
                                         scaler_u=scaler_u, 
                                         scaler_y=scaler_y)

        elif config['method'] == 'deep vanilla':
            logger = MLFlowLogger(run_id=run_id)
            model = train_deep_vanilla(normalized_train_set, normalized_val_set, config, logger)
            model_wrapper = DeepModelWrapper(model=model, 
                                             state_col=reorded_state_col, 
                                             input_col=input_col,
                                             output_col=output_col, 
                                             scaler_x=scaler_x, 
                                             scaler_u=scaler_u, 
                                             scaler_y=scaler_y)

        elif config['method'] == 'deep constraint':
            pass

        mlflow.pyfunc.log_model(name="final_model", python_model=model_wrapper)
        logging.info("Model training completed and logged to MLflow.")

if __name__ == "__main__":
    main()