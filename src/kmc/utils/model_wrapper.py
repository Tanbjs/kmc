import mlflow.pyfunc
import numpy as np
from typing import Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import torch

from ..model import DMDc, EDMDc, LitKAEc

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, 
                 model: Union[DMDc, EDMDc],
                 scaler_x: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler],
                 scaler_u: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]):
        """
        Args:
            model: The trained System ID model (DMDc/EDMDc).
            scaler_x: Scaler for state variables (x).
            scaler_u: (Optional) Scaler for control inputs (u).
        """
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_u = scaler_u

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
        u_input = self.scaler_u.transform(u_raw)
        x_next_scaled = self.model.predict(x_scaled, u_input)
        x_next = self.scaler_x.inverse_transform(x_next_scaled)

        return x_next
    
class DeepModelWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, 
                 model: LitKAEc, 
                 scaler_x: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler],
                 scaler_u: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler]):
        
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_u = scaler_u
        
        self.model.eval()
        self.model.freeze() 

    def predict(self, context, model_input):
        """
        model_input: Dictionary containing 'x' and 'u'.
        """
        x_raw = np.array(model_input['x'])
        u_raw = np.array(model_input['u'])
        if x_raw.ndim == 1: x_raw = x_raw.reshape(1, -1)
        if u_raw.ndim == 1: u_raw = u_raw.reshape(1, -1)
        x_scaled = self.scaler_x.transform(x_raw)
        u_input = self.scaler_u.transform(u_raw)
        device = self.model.device 
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        u_tensor = torch.tensor(u_input, dtype=torch.float32).to(device)
        with torch.no_grad():
            x_next_tensor, _ = self.model(x_tensor, u_tensor)
        x_next_scaled = x_next_tensor.cpu().numpy()
        x_next = self.scaler_x.inverse_transform(x_next_scaled)

        return x_next