import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from ..base import BaseObservable

class EDMDc(BaseEstimator):
    """
    Extended Dynamic Mode Decomposition with Control (EDMDc).
    
    Modified: Enforces C = [I 0], assuming the observable function 
    includes the original state x as the first components of z.
    
    Dynamics:
        z_{k+1} = A * z_k + B * u_k
    """

    def __init__(self, 
                 obs: BaseObservable, 
                 method: str = 'ols', 
                 alpha: float = 1e-5):
        super().__init__() 
        self._obs_func = obs
        self.method = method
        self.alpha = alpha
        
        # Learned System Matrices
        self.A = None  # Latent state transition (n_lifted, n_lifted)
        self.B = None  # Latent control matrix (n_lifted, n_controls)
        self.Omega = None  # Observable function and input matrix (n_lifted + n_controls)

    def fit(self, X1: np.ndarray, X2: np.ndarray, U: np.ndarray) -> 'EDMDc':
        """
        Fit the EDMDc model.
        """
        X_current = np.asarray(X1)
        X_next = np.asarray(X2)
        U_current = np.asarray(U)

        # 1. Fit & Transform Observable (Lifting)
        Z_current = self._obs_func.fit_transform(X_current)
        Z_next = self._obs_func.transform(X_next)

        # 2. Construct C Matrix as [I 0]
        # We assume z = [x, phi(x)], so x is just the first n_physical elements of z.
        n_lifted = Z_current.shape[1]

        # 3. Formulate EDMDc Linear Problem
        # Minimize || Z_next - (Z_curr * A^T + U_curr * B^T) ||^2
        # Construct Feature Matrix Omega = [Z, U]
        self.Omega = np.hstack((Z_current, U_current))
        
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
        regressor.fit(self.Omega, Z_next)
        
        # Extract A, B from the learned coefficients
        K = regressor.coef_
        
        # K shape is (n_targets, n_features) -> (n_lifted, n_lifted + n_controls)
        self.A = K[:, :n_lifted]
        self.B = K[:, n_lifted:]

        return self

    def predict(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Predict the next PHYSICAL state x_{k+1}.
        """
        self._check_fitted()
        Z = self._obs_func.fit_transform(X)
        Z_next = Z @ self.A.T + U @ self.B.T

        return Z_next

    def _check_fitted(self):
        if self.A is None or self.B is None:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted.")