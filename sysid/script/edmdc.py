import os, mlflow, argparse, logging, urllib3, dotenv
dotenv.load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import optuna as opt

from kmc.model import EDMDc
from kmc.utils.observable import PolynomialObservable, CombinedObservable, TrigonometricObservable
from utils.data import prepare_data

def train_edmdc(normalized_train_set, config):
    
    Xk_train, Uk_train, Xk1_train = prepare_data(normalized_train_set)
    logging.info(f"Xk1_train.shape: {Xk1_train.shape}, Uk_train.shape: {Uk_train.shape}, Xk_train.shape: {Xk_train.shape}")
    logging.info("Training EDMDc model...")
    
    # Define observables
    observable = config.get('observable', {})
    if observable['type'] == 'polynomial':
        degree = observable.get('degree', 2)
        poly_obs = PolynomialObservable(degree=degree, include_bias=False)

    # Fit EDMDc model
    Omega = np.hstack((np.hstack([obs.fit_transform(Xk_train) for obs in [poly_obs]]), Uk_train))
    cond_number = np.linalg.cond(Omega.T @ Omega)
    logging.info(f"Condition number of EDMDc feature matrix (before regularization): {cond_number}")
    mlflow.log_param("edmdc_feature_condition_number", cond_number)

    model = EDMDc(obs=poly_obs)
    method = config.get('options', {}).get('method')
    kwargs = config.get('options', {}).get('regression_args', {})
    model.fit(Xk_train, Xk1_train, Uk_train, method=method, **kwargs)
    logging.info(f"Using {observable['type']} observable with degree {degree} for EDMDc.")
    logging.info(f"obervable feature names: {model._obs_func.get_output_names()}")
    return model

def tune(normalized_train_set, config):
    pass

def main(*args, **kwargs):
    data = kwargs.get('data')
    config = kwargs.get('config')
    train_edmdc(normalized_train_set=data, config=config)

if __name__ == "__main__":
    main()