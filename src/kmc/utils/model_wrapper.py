import mlflow.pyfunc
import numpy as np
from typing import Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import torch

from ..model import DMDc, EDMDc, LitKAEc

class DMDcWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, 
                 model: DMDc,
                 state_col: list[str],
                 input_col: list[str],
                 output_col: list[str],
                 scaler_x: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler],
                 scaler_u: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler],
                 scaler_y: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]):
        """
        Args:
            model: The trained System ID model (DMDc/EDMDc).
            scaler_x: Scaler for state variables (x).
            scaler_u: (Optional) Scaler for control inputs (u).
        """
        self.model = model

        if list(scaler_x.feature_names_in_) != state_col:
            raise ValueError("Input columns do not match scaler_x feature names.")
        if list(scaler_u.feature_names_in_) != input_col:
            raise ValueError("Input columns do not match scaler_u feature names.")
        if list(scaler_y.feature_names_in_) != output_col:
            raise ValueError("Output columns do not match scaler_y feature names.")
        
        self.state_col = state_col
        self.input_col = input_col
        self.output_col = output_col

        self.scaler_x = scaler_x
        self.scaler_u = scaler_u
        self.scaler_y = scaler_y

        self.A = model.A
        self.B = model.B
        self.C = np.eye(len(output_col))
        
    def predict(self, context, model_input):
        """
        model_input: Dictionary containing 'x' and 'u' keys.
                     Values can be list, numpy array, or single float.
        """
        x_raw = np.array(model_input['x'])
        u_raw = np.array(model_input['u'])

        if x_raw.ndim == 1: x_raw = x_raw.reshape(1, -1)
        if u_raw.ndim == 1: u_raw = u_raw.reshape(1, -1)
        x_scaled = self.scaler_x.transform(x_raw)
        u_scaled = self.scaler_u.transform(u_raw)
        
        z_next_scaled = self.model.predict(x_scaled, u_scaled)
        y_next_scaled = z_next_scaled @ self.C.T
        
        y_next = self.scaler_y.inverse_transform(y_next_scaled)
        
        return y_next
    
class EDMDcWrapper(mlflow.pyfunc.PythonModel):    
    def __init__(self, 
                 model: EDMDc,
                 state_col: list[str],
                 input_col: list[str],
                 output_col: list[str],
                 scaler_x: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler],
                 scaler_u: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler],
                 scaler_y: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]):
        """
        Args:
            model: The trained System ID model (DMDc/EDMDc).
            scaler_x: Scaler for state variables (x).
            scaler_u: (Optional) Scaler for control inputs (u).
        """
        self.model = model

        if list(scaler_x.feature_names_in_) != state_col:
            raise ValueError("Input columns do not match scaler_x feature names.")
        if list(scaler_u.feature_names_in_) != input_col:
            raise ValueError("Input columns do not match scaler_u feature names.")
        if list(scaler_y.feature_names_in_) != output_col:
            raise ValueError("Output columns do not match scaler_y feature names.")
        
        self.state_col = state_col
        self.input_col = input_col
        self.output_col = output_col

        self.scaler_x = scaler_x
        self.scaler_u = scaler_u
        self.scaler_y = scaler_y

        self.A = model.A
        self.B = model.B
        koopman_dim = model.A.shape[0]
        self.C = np.concatenate([np.eye(len(output_col)), np.zeros((len(output_col), koopman_dim - len(output_col)))], axis=1)
        
    def predict(self, context, model_input):
        """
        model_input: Dictionary containing 'x' and 'u' keys.
                     Values can be list, numpy array, or single float.
        """
        x_raw = np.array(model_input['x'])
        u_raw = np.array(model_input['u'])

        if x_raw.ndim == 1: x_raw = x_raw.reshape(1, -1)
        if u_raw.ndim == 1: u_raw = u_raw.reshape(1, -1)
        x_scaled = self.scaler_x.transform(x_raw)
        u_scaled = self.scaler_u.transform(u_raw)
        
        z_next_scaled = self.model.predict(x_scaled, u_scaled)
        x_next_scaled = z_next_scaled[:, :len(self.state_col)]
        y_next_scaled = z_next_scaled @ self.C.T

        x_next = self.scaler_x.inverse_transform(x_next_scaled)
        y_next = self.scaler_y.inverse_transform(y_next_scaled)
        
        return x_next, y_next
    
class DeepModelWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, 
                 model: LitKAEc, 
                 state_col: list[str],
                 input_col: list[str],
                 output_col: list[str],
                 scaler_x: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler],
                 scaler_u: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler],
                 scaler_y: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]):
        
        self.model = model

        if list(scaler_x.feature_names_in_) != state_col:
            raise ValueError("Input columns do not match scaler_x feature names.")
        if list(scaler_u.feature_names_in_) != input_col:
            raise ValueError("Input columns do not match scaler_u feature names.")
        if list(scaler_y.feature_names_in_) != output_col:
            raise ValueError("Output columns do not match scaler_y feature names.")

        self.state_col = state_col
        self.input_col = input_col
        self.output_col = output_col

        self.scaler_x = scaler_x
        self.scaler_u = scaler_u
        self.scaler_y = scaler_y
        
        self.A = model.model.A.weight.detach().cpu().numpy()
        self.B = model.model.B.weight.detach().cpu().numpy()
        self.C = np.concatenate([np.eye(len(self.output_col)), np.zeros((len(self.output_col), self.model.model.total_latent_dim - len(self.output_col)))], axis=1)
        
        self.model.eval()
        self.model.freeze() 

    def update_selector(self, target_indices: list[int]):
        """Engineering Hook: Allow runtime reconfiguration for validation"""
        n_outputs = len(target_indices)
        total_latent_dim = self.model.model.total_latent_dim        
        self.C = np.zeros((n_outputs, total_latent_dim), dtype=np.float32)
        self.C[np.arange(n_outputs), target_indices] = 1.0

    def predict(self, context, model_input):
        """
        model_input: Dictionary containing 'x' and 'u'.
        """
        x_raw = np.array(model_input['x'])
        u_raw = np.array(model_input['u'])
        if x_raw.ndim == 1: x_raw = x_raw.reshape(1, -1)
        if u_raw.ndim == 1: u_raw = u_raw.reshape(1, -1)
        x_scaled = self.scaler_x.transform(x_raw)
        u_scaled = self.scaler_u.transform(u_raw)
        device = self.model.device
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        u_tensor = torch.tensor(u_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            x_next_scaled, z_next_scaled = self.model(x_tensor, u_tensor)
        z_next_scaled = z_next_scaled.cpu().numpy()
        y_next_scaled = z_next_scaled @ self.C.T

        x_next = self.scaler_x.inverse_transform(x_next_scaled)
        y_next = self.scaler_y.inverse_transform(y_next_scaled)

        return x_next, y_next