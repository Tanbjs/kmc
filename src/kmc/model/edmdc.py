import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from ..base import BaseObservable

class EDMDc(BaseEstimator):
    """
    Extended Dynamic Mode Decomposition with Control (EDMDc).
    
    This class learns a linear approximation of a nonlinear system in a lifted 
    feature space (defined by 'obs').
    
    Dynamics:
        z_{k+1} = A * z_k + B * u_k
        x_{k}   = C * z_k  (Projection from latent space back to physical space)
    
    where z = obs.transform(x).
    """

    def __init__(self, 
                 obs: BaseObservable, 
                 method: str = 'ridge', 
                 alpha: float = 1e-5):
        """
        Args:
            obs: The observable/basis function transformer (e.g., Polynomial, RBF).
            method: Regression method ['ols', 'ridge', 'lasso'].
            alpha: Regularization strength (for ridge/lasso).
        """
        super().__init__() 
        self._obs_func = obs
        self.method = method
        self.alpha = alpha
        
        # Learned System Matrices
        self.A = None  # Latent state transition (n_lifted, n_lifted)
        self.B = None  # Latent control matrix (n_lifted, n_controls)
        self.C = None  # Projection matrix z -> x (n_physical, n_lifted)

    def fit(self, X1: np.ndarray, X2: np.ndarray, U: np.ndarray) -> 'EDMDc':
        """
        Fit the EDMDc model.
        
        Process:
        1. Lift physical states X to high-dimensional space Z.
        2. Learn projection matrix C (to map Z back to X).
        3. Learn system matrices A and B in the Z-space.
        """
        X_current = np.asarray(X1)
        X_next = np.asarray(X2)
        U_current = np.asarray(U)

        # 1. Fit & Transform Observable (Lifting)
        # We refit the observable to ensure it covers the current data distribution
        self._obs_func.fit(X_current) 
        Z_current = self._obs_func.transform(X_current)
        Z_next = self._obs_func.transform(X_next)

        # 2. Find C Matrix (Projection z -> x)
        # We need a way to map the lifted state z back to the physical state x.
        # Problem: Minimize || X - Z @ C.T ||^2
        # C shape: (n_physical, n_lifted)
        
        if self.method == 'ridge':
            # Use Ridge for stability if the basis is ill-conditioned
            c_solver = Ridge(alpha=self.alpha, fit_intercept=False)
            c_solver.fit(Z_current, X_current)
            self.C = c_solver.coef_ 
        else:
            # Use OLS (Least Squares) for standard reconstruction
            # np.linalg.lstsq returns tuple, [0] is the solution
            self.C = np.linalg.lstsq(Z_current, X_current, rcond=None)[0].T

        # 3. Formulate EDMDc Linear Problem
        # Minimize || Z_next - (Z_curr * A^T + U_curr * B^T) ||^2
        
        # Construct Feature Matrix Omega = [Z, U]
        Omega = np.hstack((Z_current, U_current))
        Y = Z_next
        
        # --- REGRESSION METHOD SELECTION ---
        if self.method == 'ridge':
            regressor = Ridge(alpha=self.alpha, fit_intercept=False)
        elif self.method == 'ols':
            regressor = LinearRegression(fit_intercept=False)
        elif self.method == 'lasso':
            regressor = Lasso(alpha=self.alpha, fit_intercept=False)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'ridge', 'ols', or 'lasso'.")
            
        # 4. Solve for Operator K = [A, B]
        regressor.fit(Omega, Y)
        
        # Extract A, B from the learned coefficients
        # K shape: (n_lifted, n_lifted + n_controls)
        K = regressor.coef_
        
        n_lifted = Z_current.shape[1]
        self.A = K[:, :n_lifted]
        self.B = K[:, n_lifted:]

        return self

    def predict(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Predict the next PHYSICAL state x_{k+1}.
        
        Pipeline: x_k -> z_k -> z_{k+1} -> x_{k+1}
        """
        self._check_fitted()

        # 1. Lift: Map physical state to feature space
        Z = self._obs_func.transform(X)
        
        # 2. Evolve: Linear dynamics in feature space
        # z_{k+1} = A * z_k + B * u_k
        Z_next = Z @ self.A.T + U @ self.B.T

        # 3. Project: Map back to physical space using C
        # x_{k+1} = C * z_{k+1}
        return Z_next @ self.C.T
    
    def predict_latent(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Helper: Returns the next LATENT state z_{k+1}. 
        Useful for analysis or LMPC in latent space.
        """
        self._check_fitted()
        Z = self._obs_func.transform(X)
        return Z @ self.A.T + U @ self.B.T

    def _check_fitted(self):
        if self.A is None or self.B is None or self.C is None:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted.")