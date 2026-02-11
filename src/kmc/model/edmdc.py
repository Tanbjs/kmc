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
        x_{k}   = C * z_k  (where C = [I 0])
    """

    def __init__(self, 
                 obs: list[BaseObservable], 
                 method: str = 'ridge', 
                 alpha: float = 1e-5):
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
        """
        X_current = np.asarray(X1)
        X_next = np.asarray(X2)
        U_current = np.asarray(U)

        # 1. Fit & Transform Observable (Lifting)
        for obs in self._obs_func:
            obs.fit(X_current)
        Z_current = np.hstack([obs.transform(X_current) for obs in self._obs_func])
        Z_next = np.hstack([obs.transform(X_next) for obs in self._obs_func])

        # --- MODIFIED SECTION START ---
        # 2. Construct C Matrix as [I 0]
        # We assume z = [x, phi(x)], so x is just the first n_physical elements of z.
        
        n_physical = X_current.shape[1]
        n_lifted = Z_current.shape[1]

        if n_lifted < n_physical:
            raise ValueError(f"Lifted dimension ({n_lifted}) is smaller than physical dimension ({n_physical}). Cannot enforce C=[I 0].")

        # Create C = [I  0]
        self.C = np.zeros((n_physical, n_lifted))
        self.C[:n_physical, :n_physical] = np.eye(n_physical)

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

        # 1. Lift
        Z = np.hstack([obs.transform(X) for obs in self._obs_func])
        
        # 2. Evolve in Latent Space
        # z_{k+1} = A * z_k + B * u_k
        Z_next = Z @ self.A.T + U @ self.B.T

        # 3. Project back using C = [I 0]
        # This effectively slices the first n_physical components
        return Z_next @ self.C.T
    
    def predict_latent(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        self._check_fitted()
        Z = np.hstack([obs.transform(X) for obs in self._obs_func])
        return Z @ self.A.T + U @ self.B.T

    def _check_fitted(self):
        if self.A is None or self.B is None or self.C is None:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted.")