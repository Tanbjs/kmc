from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from kmc.utils.model_wrapper import DMDcWrapper, EDMDcWrapper, DeepModelWrapper
import mlflow

class Report():

    def __init__(self, A, B, output_col):
        self.A = A
        self.B = B
        self.output_col = output_col

    def __create_plot(self, time, y_true, y_pred, title):
        """
        Create a professional comparison plot with RMSE metrics.
        """
        num_states = y_true.shape[1]
        num_cols = 2
        num_rows = (num_states + 1) // num_cols
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3.5 * num_rows), dpi=300)
        axs_flat = axs.flatten() if num_states > 1 else [axs]
        clean_title = title.split('/')[-1] if '/' in title else title
        fig.suptitle(f"System ID Validation: {clean_title}", fontsize=16, fontweight='bold', y=0.98)

        for i in range(num_states):
            ax = axs_flat[i]
            state_name = self.output_col[i] if i < len(self.output_col) else f"State {i}"
            ax.plot(time, y_true[:, i], label='True', color="#ff3b0a", linestyle='--', linewidth=1.5, alpha=0.8)
            ax.plot(time, y_pred[:, i], label='Pred', color="#1649f3", linestyle='-', linewidth=1.5, alpha=0.9)
            ax.set_xlabel("Time Step", fontsize=9)
            ax.set_ylabel(state_name, fontsize=9)  
            ax.set_title(state_name, fontsize=11, fontweight='bold') 
            ax.grid(True, which='both', linestyle=':', alpha=0.6)
            ax.legend(loc='upper right', fontsize=9)
        for j in range(num_states, len(axs_flat)):
            fig.delaxes(axs_flat[j])
        plt.tight_layout(rect=[0, 0.0, 1, 0.96])
        
        return fig
    
    def log_plot(self, test_results: List[tuple], n: int):
        
        # plot time series for each test file
        for filename, y_true, y_one_step_pred, y_n_step_pred in test_results:
            time = np.arange(y_true.shape[0]) 
            title_one_step = f"One-step Prediction - {filename}"
            fig_one_step = self.__create_plot(time, y_true, y_one_step_pred, title_one_step)
            mlflow.log_figure(fig_one_step, artifact_file=f"fig/one-step/{filename}_one_step_prediction.svg")
            plt.close(fig_one_step)

            title_n_step = f"{n}-step Prediction - {filename}"
            fig_n_step = self.__create_plot(time, y_true, y_n_step_pred, title_n_step)
            mlflow.log_figure(fig_n_step, artifact_file=f"fig/n-step/{filename}_{n}_step_prediction.svg")
            plt.close(fig_n_step)

        # plot eigenvalues
        eigenvalues = np.linalg.eigvals(self.A)
        max_real = np.max(eigenvalues.real)
        fig, ax = plt.subplots(figsize=(6,6), dpi=300)
        unit_circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1)
        ax.add_artist(unit_circle)
        ax.scatter(eigenvalues.real, eigenvalues.imag, color='#1649f3', s=50, label='Eigenvalues')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlabel('Real Part', fontsize=10)
        ax.set_ylabel('Imaginary Part', fontsize=10)
        ax.set_title(f'Eigenvalue Spectrum (Max Real: {max_real:.8f})', fontsize=12, fontweight='bold')
        ax.grid(True, which='both', linestyle=':', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.7)
        ax.axvline(0, color='black', linewidth=0.8, alpha=0.7)
        plt.tight_layout()
        mlflow.log_figure(fig, artifact_file='fig/eigenvalues/eigenvalue_spectrum.svg')
        plt.close(fig)
    
    def log_score(self, test_results: List[tuple]):
        
        score = []

        for filename, y_true, y_one_step_pred, y_n_step_pred in test_results:

            # Reference Deviation 
            y_mean = np.mean(y_true, axis=0)
            ref_deviation = y_true - y_mean
            norm_ref = np.linalg.norm(ref_deviation, axis=0) + 1e-9 
            sq_norm_ref = norm_ref ** 2
            err_one = y_true - y_one_step_pred
            err_n   = y_true - y_n_step_pred
            
            def calculate_metrics(error, norm_r, sq_norm_r):
                mse = np.mean(error**2, axis=0)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(error), axis=0)
                max_ae = np.max(np.abs(error), axis=0)
                
                # Cost Functions
                norm_e = np.linalg.norm(error, axis=0)
                sq_norm_e = norm_e ** 2
                
                # NRMSE & NMSE Costs
                nrmse = norm_e / norm_r
                nmse  = sq_norm_e / sq_norm_r 
                
                r2 = 1 - nmse
                return r2, nrmse, nmse, mse, rmse, mae, max_ae

            # Get Metrics for One-Step
            r2_1, nrmse_1, nmse_1, mse_1, rmse_1, mae_1, max_1 = calculate_metrics(err_one, norm_ref, sq_norm_ref)
            
            # Get Metrics for N-Step
            r2_n, nrmse_n, nmse_n, mse_n, rmse_n, mae_n, max_n = calculate_metrics(err_n, norm_ref, sq_norm_ref)

            score_one_step_df = pd.DataFrame({
                'state': self.output_col,
                'r2 (%)': r2_1 * 100,       # Convert to percentage
                'nrmse': nrmse_1,           # Recommended for Control
                'nmse': nmse_1,             # R2 in %
                'mse': mse_1,               # Raw MSE
                'rmse': rmse_1,
                'mae': mae_1,
                'maxae': max_1
            })
            
            score_N_step_df = pd.DataFrame({
                'state': self.output_col,
                'r2 (%)': r2_n * 100,        # Convert to percentage
                'nrmse': nrmse_n,            # Recommended for Control
                'nmse': nmse_n,              # R2 in %
                'mse': mse_n,                # Raw MSE
                'rmse': rmse_n,
                'mae': mae_n,
                'maxae': max_n
            })

            score.append((filename, score_one_step_df, score_N_step_df))

        # Average score across all test files
        all_one_step = [item[1] for item in score] 
        all_n_step   = [item[2] for item in score]
        
        # GroupBy & Mean
        avg_score_one_step = pd.concat(all_one_step).groupby('state', sort=False).mean().reset_index()
        avg_score_N_step   = pd.concat(all_n_step).groupby('state', sort=False).mean().reset_index()

        # Log to MLflow
        mlflow.log_table(avg_score_one_step, artifact_file='avg_score_one_step.json')
        mlflow.log_table(avg_score_N_step, artifact_file=f'avg_score_{self.n}_step.json')

    def controlability_check(self):
        n = self.A.shape[0]
        ctr_mat = self.B
        for i in range(1, n):
            ctr_mat = np.hstack((ctr_mat, self.A @ ctr_mat[:, -self.B.shape[1]:]))
        rank_ctr = np.linalg.matrix_rank(ctr_mat)
        is_controllable = (rank_ctr == n)
        mlflow.log_param("is_controllable", is_controllable)
        mlflow.log_param("controllability_rank", rank_ctr)

        if is_controllable:
            mlflow.log_param("is_stabilizable", True)
        else:
            eigenvalues, _ = np.linalg.eig(self.A)
            unstable_eigenvalues = eigenvalues[np.abs(eigenvalues) >= 1.0 - 1e-6] # tolerance
            is_stabilizable = True
            for lam in unstable_eigenvalues:
                # PBH Matrix: [ A - lambda*I | B ]
                pbh_matrix = np.hstack((self.A - lam * np.eye(n), self.B))
                if np.linalg.matrix_rank(pbh_matrix) < n:
                    is_stabilizable = False
                    break
            mlflow.log_param("is_stabilizable", is_stabilizable)

class DMDcPredictor(Report):

    def __init__(self, model: DMDcWrapper, n_test):
        self.model = model
        self.n = n_test
        self.test_results = []
        super().__init__(model.A, model.B, model.output_col)
        
    def log_plot(self):
        super().log_plot(self.test_results, self.n)

    def log_score(self):
        super().log_score(self.test_results)
    
    def controlability_check(self):
        super().controlability_check()

    def predict(self, test_set: List[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]):
        for filename, state_df, input_df, output_df in test_set:
            x_curr = state_df.iloc[:-1].to_numpy()
            u_curr = input_df.iloc[:-1].to_numpy()
            y_curr = output_df.iloc[:-1].to_numpy()
            y_one_step_pred = self.__one_step_prediction(x_curr, u_curr)
            y_n_step_pred = self.__n_step_prediction(x_curr, u_curr)
            self.test_results.append((filename, y_curr, y_one_step_pred, y_n_step_pred))
        
        return self

    def __one_step_prediction(self, x_curr, u_curr):
        y_one_step_pred = np.zeros((len(x_curr), len(self.model.output_col)))
        for i in range(u_curr.shape[0]):
            y_next = self.model.predict(context=None, model_input={'x': x_curr[i,:].reshape(1, -1), 'u': u_curr[i,:].reshape(1, -1)})
            y_one_step_pred[i,:] = y_next.squeeze()

        return y_one_step_pred

    def __n_step_prediction(self, x_curr, u_curr):
        num_samples = len(u_curr)
        y_n_step_pred = np.zeros((num_samples, len(self.model.output_col)))
        x_k = x_curr[0, :].reshape(1, -1)
        for i in range(num_samples):
            if i % self.n == 0:
                x_k = x_curr[i, :].reshape(1, -1)
            y_next = self.model.predict(context=None, model_input={'x': x_k, 'u': u_curr[i, :].reshape(1, -1)}) 
            y_n_step_pred[i, :] = y_next.squeeze()
            x_k = y_next.reshape(1, -1)

        return y_n_step_pred
    
class EDMDcPredictor(Report):

    def __init__(self, model: EDMDcWrapper, n_test):
        self.model = model
        self.n = n_test
        self.test_results = []
        super().__init__(model.A, model.B, model.output_col)
    
    def log_plot(self):
        super().log_plot(self.test_results, self.n)

    def log_score(self):
        super().log_score(self.test_results)
    
    def controlability_check(self):
        super().controlability_check()

    def predict(self, test_set: List[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]):

        for filename, state_df, input_df, output_df in test_set:
            x_curr = state_df.iloc[:-1].to_numpy()
            u_curr = input_df.iloc[:-1].to_numpy()
            y_curr = output_df.iloc[:-1].to_numpy()
            y_one_step_pred = self.__one_step_prediction(x_curr, u_curr)
            y_n_step_pred = self.__n_step_prediction(x_curr, u_curr)
            self.test_results.append((filename, y_curr, y_one_step_pred, y_n_step_pred))
        
        return self

    def __one_step_prediction(self, x_curr, u_curr):
        y_one_step_pred = np.zeros((len(x_curr), len(self.model.output_col)))
        for i in range(u_curr.shape[0]):
            _, y_next = self.model.predict(context=None, model_input={'x': x_curr[i,:].reshape(1, -1), 'u': u_curr[i,:].reshape(1, -1)})
            y_one_step_pred[i,:] = y_next.squeeze()

        return y_one_step_pred

    def __n_step_prediction(self, x_curr, u_curr):
        num_samples = len(u_curr)
        y_n_step_pred = np.zeros((num_samples, len(self.model.output_col)))
        x_k = x_curr[0, :].reshape(1, -1)
        for i in range(num_samples):
            if i % self.n == 0:
                x_k = x_curr[i, :].reshape(1, -1)
            x_next, y_next = self.model.predict(context=None, model_input={'x': x_k, 'u': u_curr[i, :].reshape(1, -1)})
            y_n_step_pred[i, :] = y_next.squeeze()
            x_k = x_next.reshape(1, -1) 

        return y_n_step_pred

class DeepPredictor(Report):
    
    def __init__(self, model: DeepModelWrapper, n_test):
        self.model = model
        self.test_results = []
        self.n = n_test
        super().__init__(model.A, model.B, model.output_col)

    def log_plot(self):
        super().log_plot(self.test_results, self.n)

    def log_score(self):
        super().log_score(self.test_results)
    
    def controlability_check(self):
        super().controlability_check()

    def predict(self, test_set: List[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]]):

        for filename, state_df, input_df, output_df in test_set:
            x_curr = state_df.iloc[:-1].to_numpy()
            u_curr = input_df.iloc[:-1].to_numpy()
            y_curr = output_df.iloc[:-1].to_numpy()
            y_one_step_pred = self.__one_step_prediction(x_curr, u_curr)
            y_n_step_pred = self.__n_step_prediction(x_curr, u_curr)
            self.test_results.append((filename, y_curr, y_one_step_pred, y_n_step_pred))

        return self
    
    def __one_step_prediction(self, x_curr, u_curr):
        y_one_step_pred = np.zeros((len(x_curr), len(self.model.output_col)))
        for i in range(u_curr.shape[0]):
            _, y_next = self.model.predict(context=None, model_input={'x': x_curr[i,:].reshape(1, -1), 'u': u_curr[i,:].reshape(1, -1)})
            y_one_step_pred[i,:] = y_next.squeeze()

        return y_one_step_pred

    def __n_step_prediction(self, x_curr, u_curr):
        num_samples = len(u_curr)
        y_n_step_pred = np.zeros((num_samples, len(self.model.output_col)))
        x_k = x_curr[0, :].reshape(1, -1)
        for i in range(num_samples):
            if i % self.n == 0:
                x_k = x_curr[i, :].reshape(1, -1)
            x_next, y_next = self.model.predict(context=None, model_input={'x': x_k, 'u': u_curr[i, :].reshape(1, -1)})
            y_n_step_pred[i, :] = y_next.squeeze()
            x_k = x_next.reshape(1, -1) 

        return y_n_step_pred